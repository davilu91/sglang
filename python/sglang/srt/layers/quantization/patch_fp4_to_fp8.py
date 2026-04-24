"""
Patch sglang's Fp8MoEMethod to dequant MXFP4 -> FP8 at load time for non-Blackwell GPUs.
This enables E2E testing of DeepSeek-V4 on H20 GPUs.
"""
import torch
import logging

logger = logging.getLogger(__name__)

# FP4 E2M1 lookup table (4-bit float: 1 sign + 2 exp + 1 mantissa)
FP4_E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def dequant_mxfp4_to_bf16(weight_int8, scale, block_k=32):
    """
    Dequantize MXFP4 packed int8 weights to BF16.
    weight_int8: [N, K//2] int8 (2 FP4 values per byte, low nibble first)
    scale: [N, ceil(K/block_k)] - can be float8_e8m0fnu OR float32 (if already decoded by weight loader)
    Returns: [N, K] bfloat16
    """
    device = weight_int8.device
    N, K_half = weight_int8.shape
    K = K_half * 2

    # Unpack nibbles
    raw = weight_int8.to(torch.uint8)
    low = (raw & 0x0F).to(torch.long)
    high = ((raw >> 4) & 0x0F).to(torch.long)
    unpacked = torch.stack([low, high], dim=-1).reshape(N, K)

    # Lookup FP4 values
    table = FP4_E2M1_TABLE.to(device)
    values = table[unpacked]  # [N, K] float32

    # Decode scales - handle both E8M0 and already-decoded float32
    if scale.dtype == torch.float32 or scale.dtype == torch.bfloat16 or scale.dtype == torch.float16:
        # Scale already decoded by weight loader (E8M0 -> float32 via copy_)
        scale_float = scale.float()
    else:
        # Raw E8M0: decode as 2^(bits - 127)
        scale_bits = scale.view(torch.uint8).to(torch.int32)
        scale_float = torch.pow(2.0, (scale_bits.float() - 127.0))

    # Expand scales to match K dimension
    n_blocks = scale_float.shape[1]
    scale_expanded = scale_float.unsqueeze(-1).expand(N, n_blocks, block_k).reshape(N, n_blocks * block_k)
    scale_expanded = scale_expanded[:, :K]

    result = (values * scale_expanded).to(torch.bfloat16)
    return result


def quantize_bf16_to_fp8_block(tensor_bf16, block_n=128, block_k=128):
    """
    Quantize BF16 tensor to block FP8 E4M3.
    Returns: (weight_fp8 [N, K] float8_e4m3fn, scale_inv [N//block_n, K//block_k] float32)
    """
    N, K = tensor_bf16.shape
    pad_n = (block_n - N % block_n) % block_n
    pad_k = (block_k - K % block_k) % block_k
    if pad_n > 0 or pad_k > 0:
        tensor_bf16 = torch.nn.functional.pad(tensor_bf16, (0, pad_k, 0, pad_n))

    Np, Kp = tensor_bf16.shape
    blocks = tensor_bf16.reshape(Np // block_n, block_n, Kp // block_k, block_k)
    blocks = blocks.permute(0, 2, 1, 3).contiguous()

    amax = blocks.abs().amax(dim=(-2, -1)).float().clamp(min=1e-12)
    fp8_max = 448.0
    scale_inv = amax / fp8_max

    blocks_scaled = blocks.float() / scale_inv.unsqueeze(-1).unsqueeze(-1)
    blocks_scaled = blocks_scaled.clamp(-fp8_max, fp8_max)
    result = blocks_scaled.permute(0, 2, 1, 3).reshape(Np, Kp)
    result = result[:N, :K]

    sn = (N + pad_n) // block_n
    sk = (K + pad_k) // block_k
    weight_fp8 = result.to(torch.float8_e4m3fn)
    return weight_fp8, scale_inv[:sn, :sk].float()


def convert_fp4_layer_to_fp8(layer):
    """Convert a single MoE layer's FP4 weights to FP8 in-place."""
    for wname, sname in [('w13_weight', 'w13_weight_scale_inv'),
                         ('w2_weight', 'w2_weight_scale_inv')]:
        w_param = getattr(layer, wname)
        s_param = getattr(layer, sname)

        w_int8 = w_param.data  # [E, N, K//2] int8
        s_data = s_param.data  # [E, N, K//block_k] - float32 (decoded) or e8m0

        E = w_int8.shape[0]
        logger.info(f"Converting {wname}: weight {w_int8.shape} {w_int8.dtype}, scale {s_data.shape} {s_data.dtype}")

        new_weights = []
        new_scales = []

        for e in range(E):
            bf16 = dequant_mxfp4_to_bf16(w_int8[e], s_data[e])
            fp8_w, fp8_s = quantize_bf16_to_fp8_block(bf16)
            new_weights.append(fp8_w)
            new_scales.append(fp8_s)

        new_w = torch.stack(new_weights)
        new_s = torch.stack(new_scales)

        w_param.data = new_w
        s_param.data = new_s
        logger.info(f"Converted {wname}: {w_int8.shape} int8 -> {new_w.shape} fp8, scales {s_data.shape} -> {new_s.shape}")

print("FP4->FP8 conversion module loaded")
