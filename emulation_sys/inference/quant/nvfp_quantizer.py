import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
# import nvfp.ops as ops
import nvfp.pseudo_quant as pseudo_quant
# from inference.quant.nvfp_kernel.emulation import EmulationKernel


_FALLBACK_WARNED = False

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _is_no_kernel_image_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "no kernel image is available for execution on the device" in msg


def _torch_nvfp4_pseudo_quantize_with_scale(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype not in (torch.float, torch.float16, torch.bfloat16):
        raise ValueError(f"x.dtype must be float/fp16/bf16, got {x.dtype}")
    if x.ndim < 1:
        raise ValueError(f"x.ndim must be >= 1, got {x.ndim}")
    if x.shape[-1] % 16 != 0:
        raise ValueError(f"last dim must be divisible by 16, got {x.shape[-1]}")

    org_shape = x.shape
    x2d = x.reshape(-1, org_shape[-1]).contiguous()

    # global scale
    amax = x2d.float().abs().amax().clamp(min=1e-12)
    global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / amax).to(torch.float32)

    # block-wise scale (block size = 16)
    block_size = 16
    x_fp32_blk = x2d.float().reshape(x2d.shape[0], -1, block_size)
    max_abs = torch.amax(torch.abs(x_fp32_blk), dim=-1)
    block_scale_fp32 = (max_abs / FLOAT4_E2M1_MAX).float()

    scaled_block_scale_fp32 = block_scale_fp32 * global_scale
    scaled_block_scale_fp8 = torch.clamp(
        scaled_block_scale_fp32,
        min=FLOAT8_E4M3_EPS,
        max=FLOAT8_E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    scaled_block_scale_fp8_fp32 = scaled_block_scale_fp8.to(torch.float32)

    # Replace reciprocal_approximate_ftz_tensor with pure torch reciprocal path.
    inv_global_scale = torch.reciprocal(global_scale)
    total_scale = scaled_block_scale_fp8_fp32 * inv_global_scale
    total_scale = total_scale.clamp(min=1e-12)

    x_scaled = x_fp32_blk * torch.reciprocal(total_scale.unsqueeze(-1))
    x_scaled = torch.clamp(x_scaled, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX)
    x_scaled = x_scaled.reshape(x2d.shape)

    # quantize to fp4 then dequantize with scale
    x_fp4 = pseudo_quant.to_fp4(x_scaled)
    x_deq = pseudo_quant.unpack_fp4_bytes(x_fp4)
    x_deq = x_deq.reshape(x2d.shape[0], -1, block_size)
    x_deq = (x_deq * total_scale.unsqueeze(-1)).reshape(x2d.shape)
    return x_deq.reshape(org_shape)


def _safe_nvfp4_pseudo_quantize(x: torch.Tensor) -> torch.Tensor:
    global _FALLBACK_WARNED
    try:
        return pseudo_quant.nvfp4_pseudo_quantize(x)
    except Exception as e:
        if _is_no_kernel_image_error(e):
            if not _FALLBACK_WARNED:
                warnings.warn(
                    "nvfp4_pseudo_quantize kernel is unavailable on this GPU; falling back to scale-aware torch NVFP4 pseudo quantization.",
                    RuntimeWarning,
                )
                _FALLBACK_WARNED = True
            return _torch_nvfp4_pseudo_quantize_with_scale(x)
        raise


class QuantLinear(nn.Module):
    def __init__(
        self,
        lin: nn.Linear,
        w_bit: int,
        a_bit: int,
        group_size: int = 32,
        use_zero_point: bool = False,
        mode: str = "pseudo",  # "pseudo" or "real"
    ):
        super().__init__()
        assert mode in ("pseudo", "real", "emulation"), "mode must be 'pseudo', 'real' or 'emulation'"

        self.in_features = lin.in_features
        self.out_features = lin.out_features
        if lin.bias is not None:
            self.register_buffer("bias", lin.bias.detach().clone())
        else:
            self.bias = None
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.dtype = lin.weight.dtype
        self.group_size = group_size
        self.use_zero_point = use_zero_point
        self.mode = mode

        with torch.no_grad():
            W = lin.weight.detach()  # [out, in]
            if self.mode == "pseudo":
                Wq = _safe_nvfp4_pseudo_quantize(W).float()
                self.register_buffer("qweight_fp", Wq)
            else:  # "real" or "emulation"
                self.FLOAT4_E2M1_MAX = 6.0
                self.FLOAT8_E4M3_MAX = 448.0
                w_amax = torch.abs(W).max().to(torch.float32)
                w_global_scale = self.FLOAT8_E4M3_MAX * self.FLOAT4_E2M1_MAX / w_amax
                w_fp4, scale_w_fp4 = ops.scaled_fp4_quant(W, w_global_scale)
                self.register_buffer("w_fp4", w_fp4)
                self.register_buffer("w_scale_fp4", scale_w_fp4)
                self.w_global_scale = w_global_scale
                self.qweight_fp = None  # not used in real mode
                
                # Create emulation kernel for deterministic modeling
                # if self.mode == "emulation":
                #     self.emulation_kernel = EmulationKernel.for_rtx_5090()

        if self.bias is not None and isinstance(self.bias, torch.Tensor):
            self.bias = self.bias.to(torch.double)

    @classmethod
    def from_linear(
        cls, lin, w_bit, a_bit, group_size=32, use_zero_point=False, mode="pseudo"
    ):
        return cls(lin, w_bit, a_bit, group_size, use_zero_point, mode)

    def forward(self, x):
        # x: [*, in_features]
        original_shape_prefix = x.shape[:-1]

        if self.mode == "pseudo":
            #目前只会走pseudo
            if self.a_bit is not None and self.a_bit < 16:
                x_q = _safe_nvfp4_pseudo_quantize(x)
                x_q = x_q.double().contiguous()
            else:
                x_q = x.to(torch.double)
            # print(F.linear(x_q, self.qweight_fp.double(), self.bias).to(x.dtype).mean().item())
            return F.linear(x_q, self.qweight_fp.double(), self.bias).to(x.dtype)
        else:  # "real" or "emulation"
            x_amax = torch.abs(x).max().to(torch.float32)
            x_global_scale = self.FLOAT8_E4M3_MAX * self.FLOAT4_E2M1_MAX / x_amax
            x_fp4, scale_x_fp4 = ops.scaled_fp4_quant(x, x_global_scale)

            alpha = 1.0 / (x_global_scale * self.w_global_scale)
            if self.mode == "real":
                output = ops.cutlass_scaled_fp4_mm(
                    x_fp4, self.w_fp4, scale_x_fp4, self.w_scale_fp4, alpha, self.dtype
                )
            else:  # "emulation"
                raise NotImplementedError(
                    "mode='emulation' is not wired in this QuantLinear yet; please use mode='pseudo' or mode='real'."
                )
            # output = self.emulation_kernel(
            # output = self.emulation_kernel(
            #     output = self.emulation_kernel(
            #         x_fp4, self.w_fp4, scale_x_fp4, self.w_scale_fp4, alpha, self.dtype
            #     )
            
            # test
            # output_real = ops.cutlass_scaled_fp4_mm(
            #         x_fp4, self.w_fp4, scale_x_fp4, self.w_scale_fp4, alpha, self.dtype
            #     )
            # if self.a_bit is not None and self.a_bit < 16:
            #     x_q = pseudo_quant.nvfp4_pseudo_quantize(x)
            #     x_q = x_q.double().contiguous()
            # else:
            #     x_q = x.to(torch.double)
            # output_pseudo = F.linear(x_q, self.qweight_fp.double(), self.bias).to(x.dtype)
            # print(output_real, (output_real-output).amax(), (output_real-output).mean(), (output_real-output_pseudo).mean())
            # print(output_real, (output_real-output).amax(), (output_real-output).mean())
            # exit(0)

            # reshape output to original batch shape
            output = output.view(*original_shape_prefix, self.out_features)

            if self.bias is not None:
                output += self.bias

            return output.to(self.dtype)
