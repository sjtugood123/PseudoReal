import torch
import argparse
import torch.utils.benchmark as benchmark

import ops

# Constants for FP4 quantization
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
ALPHA_SCALE = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX

def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

# ----------------------------
# FP4 Offline GEMM Callable
# ----------------------------
class FP4GemmWithOnlineActivation:
    def __init__(self, weight_bf16):
        """Pre-quantize weight (B) once."""
        self.weight_bf16 = weight_bf16
        w_amax = torch.abs(weight_bf16).max().to(torch.float32)
        self.w_scale = ALPHA_SCALE / w_amax
        self.w_fp4, self.s_w = ops.scaled_fp4_quant(weight_bf16, self.w_scale)

    def __call__(self, activation_bf16):
        """Quantize activation online and compute FP4 GEMM."""
        a_amax = torch.abs(activation_bf16).max().to(torch.float32)
        a_scale = ALPHA_SCALE / a_amax
        a_fp4, s_a = ops.scaled_fp4_quant(activation_bf16, a_scale)
        alpha = 1.0 / (a_scale * self.w_scale)
        return ops.cutlass_scaled_fp4_mm(a_fp4, self.w_fp4, s_a, self.s_w, alpha, torch.bfloat16)

# ----------------------------
# Baseline: BF16 GEMM
# ----------------------------
def bf16_gemm(a, b):
    return torch.mm(a, b)

def main():
    parser = argparse.ArgumentParser(description='Compare BF16 GEMM vs FP4 GEMM (B offline quantized)')
    parser.add_argument('--batch_size', type=int, default=4096, help='M dimension (batch)')
    parser.add_argument('--repeats', type=int, default=1000, help='Timing repeats')
    args = parser.parse_args()

    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    M = args.batch_size

    print(f"{'Size (K=N)':<10} {'BF16 (ms)':<12} {'FP4 (ms)':<12} {'Speedup':<10} {'BF16 TFLOPS':<14} {'FP4 TFLOPS':<12}")
    print("-" * 85)

    for size in sizes:
        K = N = size

        # Prepare tensors in bfloat16
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

        flops = 2 * M * K * N  # standard matmul FLOPs

        # ---- Baseline: BF16 GEMM ----
        for _ in range(5):
            _ = bf16_gemm(a, b)
        torch.cuda.synchronize()
        _, times_bf16 = benchmark_forward(bf16_gemm, a, b, repeats=args.repeats, verbose=False)
        time_bf16 = times_bf16.mean * 1000
        tflops_bf16 = flops / times_bf16.mean / 1e12

        # ---- FP4: B offline, A online (quant + gemm timed) ----
        fp4_gemm = FP4GemmWithOnlineActivation(b)
        for _ in range(5):
            _ = fp4_gemm(a)
        torch.cuda.synchronize()
        _, times_fp4 = benchmark_forward(fp4_gemm, a, repeats=args.repeats, verbose=False)
        time_fp4 = times_fp4.mean * 1000
        tflops_fp4 = flops / times_bf16.mean / 1e12

        speedup = time_bf16 / time_fp4 if time_fp4 > 0 else float('inf')

        print(f"{size:<10} {time_bf16:<12.3f} {time_fp4:<12.3f} {speedup:<10.2f} {tflops_bf16:<14.2f} {tflops_fp4:<12.2f}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()