import argparse
import os
import random
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM


# Keep import behavior consistent with non_reasoning.py
sys.path.append(os.path.join(os.path.dirname(__file__), "emulation_sys"))
from inference.quant.pre_quant import replace_quant_linear  # type: ignore  # noqa: E402


@dataclass
class DiffStat:
    compared_modules: int = 0
    exact_equal_modules: int = 0
    shape_mismatch_modules: int = 0
    dtype_mismatch_modules: int = 0
    missing_attr_modules: int = 0
    value_diff_modules: int = 0
    max_abs_diff_w_fp4: float = 0.0
    max_abs_diff_w_scale_fp4: float = 0.0
    max_abs_diff_w_global_scale: float = 0.0


@dataclass
class ForwardStat:
    compared_modules: int = 0
    exact_equal_modules: int = 0
    missing_attr_modules: int = 0
    runtime_error_modules: int = 0
    value_diff_modules: int = 0
    max_abs_diff_output: float = 0.0
    max_mean_abs_diff_output: float = 0.0


def _load_base_model(model_path: str, dtype: torch.dtype) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def _apply_quant_replace(model: nn.Module, mode: str) -> nn.Module:
    replace_quant_linear(
        model=model,
        w_bit=4,
        a_bit=4,
        q_config={"q_group_size": 16, "mode": mode},
        use_zero_point=False,
        init_only=False,
        nvfp=True,
        fp8=False,
    )
    model.eval()
    return model


def _as_tensor(v):
    if isinstance(v, torch.Tensor):
        return v
    if isinstance(v, (float, int)):
        return torch.tensor(v)
    return None


def _compare_tensor_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, float]:
    if a.shape != b.shape:
        return False, float("inf")

    # Important: do not cast on CUDA. Some packed/custom dtypes can trigger
    # device-side asserts during CUDA cast kernels. Move to CPU first.
    a_cpu = a.detach().contiguous().cpu()
    b_cpu = b.detach().contiguous().cpu()

    try:
        exact = torch.equal(a_cpu, b_cpu)
    except Exception:
        # Fallback for exotic/private dtypes where torch.equal may assert internally.
        # Compare raw storage bytes instead.
        try:
            a_bytes = bytes(a_cpu.untyped_storage())
            b_bytes = bytes(b_cpu.untyped_storage())
            exact = (a_cpu.shape == b_cpu.shape) and (a_cpu.stride() == b_cpu.stride()) and (a_bytes == b_bytes)
        except Exception:
            # Last resort: if we cannot introspect storage safely, mark as non-exact.
            exact = False

    if a_cpu.numel() == 0:
        return exact, 0.0

    try:
        if torch.is_floating_point(a_cpu) or torch.is_complex(a_cpu):
            diff = (a_cpu.to(torch.float64) - b_cpu.to(torch.float64)).abs().max().item()
            return exact, float(diff)
        if a_cpu.dtype == torch.bool:
            diff = (a_cpu.to(torch.int8) - b_cpu.to(torch.int8)).abs().max().item()
            return exact, float(diff)

        diff = (a_cpu.to(torch.int64) - b_cpu.to(torch.int64)).abs().max().item()
        return exact, float(diff)
    except Exception:
        # Fallback for exotic dtypes where subtraction/promotion is unsupported.
        return exact, float("nan")


def _safe_update_max(current: float, candidate: float) -> float:
    if isinstance(candidate, float) and candidate != candidate:  # NaN check
        return current
    return max(current, candidate)


def _tensor_to_uint8_bytes(t: torch.Tensor) -> torch.Tensor:
    t_cpu = t.detach().contiguous().cpu()
    if t_cpu.dtype == torch.uint8:
        return t_cpu.view(torch.uint8).flatten()
    if t_cpu.dtype == torch.float4_e2m1fn_x2:
        return t_cpu.view(torch.uint8).flatten()
    return t_cpu.view(torch.uint8).flatten()


def _pack_bytes_to_uint32_words(byte_tensor: torch.Tensor) -> list[int]:
    b = byte_tensor.tolist()
    word_count = len(b) // 4
    words: list[int] = []
    for i in range(word_count):
        j = i * 4
        w = int(b[j]) | (int(b[j + 1]) << 8) | (int(b[j + 2]) << 16) | (int(b[j + 3]) << 24)
        words.append(w)
    return words


def dump_w_fp4_uint32_bits(
    real_model: nn.Module,
    emu_model: nn.Module,
    *,
    dump_count: int = 8,
    quick_check: bool = False,
):
    emu_modules = dict(emu_model.named_modules())
    quant_module_items = [(name, mod_real) for name, mod_real in real_model.named_modules() if hasattr(mod_real, "w_fp4")]
    if quick_check:
        quant_module_items = quant_module_items[:3]

    if not quant_module_items:
        print("[bit-dump] no quantized modules found.")
        return

    target_name, mod_real = quant_module_items[0]
    mod_emu = emu_modules.get(target_name)
    if mod_emu is None or not hasattr(mod_emu, "w_fp4"):
        print(f"[bit-dump] target module missing in emu model: {target_name}")
        return

    real_w = getattr(mod_real, "w_fp4")
    emu_w = getattr(mod_emu, "w_fp4")

    real_bytes = _tensor_to_uint8_bytes(real_w)
    emu_bytes = _tensor_to_uint8_bytes(emu_w)

    real_words = _pack_bytes_to_uint32_words(real_bytes)
    emu_words = _pack_bytes_to_uint32_words(emu_bytes)

    n = min(dump_count, len(real_words), len(emu_words))
    print("\n==== uint32 bit dump (w_fp4) ====")
    print(f"module: {target_name}")
    print(f"real dtype: {real_w.dtype}, emu dtype: {emu_w.dtype}")
    print(f"showing first {n} uint32 words (little-endian packed from bytes)")
    for i in range(n):
        rw = real_words[i]
        ew = emu_words[i]
        print(
            f"[{i:03d}] real=0x{rw:08x} ({rw:032b}) | "
            f"emu=0x{ew:08x} ({ew:032b}) | equal={rw == ew}"
        )


def compare_real_vs_emulation(
    real_model: nn.Module,
    emu_model: nn.Module,
    topk: int = 20,
    quick_check: bool = False,
) -> DiffStat:
    stat = DiffStat()

    emu_modules = dict(emu_model.named_modules())
    diff_lines: list[str] = []

    quant_module_items = [(name, mod_real) for name, mod_real in real_model.named_modules() if hasattr(mod_real, "w_fp4")]

    if quick_check:
        quant_module_items = quant_module_items[:3]
        print(f"[quick-check] enabled: only comparing first {len(quant_module_items)} quantized linear modules.")

    for name, mod_real in tqdm(quant_module_items, desc="Comparing quantized modules", total=len(quant_module_items)):

        stat.compared_modules += 1

        if name not in emu_modules:
            stat.missing_attr_modules += 1
            diff_lines.append(f"[MISSING_MODULE] {name}")
            continue

        mod_emu = emu_modules[name]

        for attr in ("w_fp4", "w_scale_fp4", "w_global_scale"):
            if not hasattr(mod_real, attr) or not hasattr(mod_emu, attr):
                stat.missing_attr_modules += 1
                diff_lines.append(f"[MISSING_ATTR] {name}.{attr}")
                break
        else:
            t_real_w = _as_tensor(getattr(mod_real, "w_fp4"))
            t_emu_w = _as_tensor(getattr(mod_emu, "w_fp4"))
            t_real_s = _as_tensor(getattr(mod_real, "w_scale_fp4"))
            t_emu_s = _as_tensor(getattr(mod_emu, "w_scale_fp4"))
            t_real_g = _as_tensor(getattr(mod_real, "w_global_scale"))
            t_emu_g = _as_tensor(getattr(mod_emu, "w_global_scale"))

            if t_real_w is None or t_emu_w is None or t_real_s is None or t_emu_s is None or t_real_g is None or t_emu_g is None:
                stat.missing_attr_modules += 1
                diff_lines.append(f"[NON_TENSOR_ATTR] {name}")
                continue

            # shape / dtype checks
            if t_real_w.shape != t_emu_w.shape or t_real_s.shape != t_emu_s.shape:
                stat.shape_mismatch_modules += 1
                diff_lines.append(
                    f"[SHAPE_MISMATCH] {name}: w {tuple(t_real_w.shape)} vs {tuple(t_emu_w.shape)}, "
                    f"scale {tuple(t_real_s.shape)} vs {tuple(t_emu_s.shape)}"
                )
                continue

            if t_real_w.dtype != t_emu_w.dtype or t_real_s.dtype != t_emu_s.dtype:
                stat.dtype_mismatch_modules += 1
                diff_lines.append(
                    f"[DTYPE_MISMATCH] {name}: w {t_real_w.dtype} vs {t_emu_w.dtype}, "
                    f"scale {t_real_s.dtype} vs {t_emu_s.dtype}"
                )

            eq_w, diff_w = _compare_tensor_pair(t_real_w, t_emu_w)
            eq_s, diff_s = _compare_tensor_pair(t_real_s, t_emu_s)
            eq_g, diff_g = _compare_tensor_pair(t_real_g, t_emu_g)

            stat.max_abs_diff_w_fp4 = _safe_update_max(stat.max_abs_diff_w_fp4, diff_w)
            stat.max_abs_diff_w_scale_fp4 = _safe_update_max(stat.max_abs_diff_w_scale_fp4, diff_s)
            stat.max_abs_diff_w_global_scale = _safe_update_max(stat.max_abs_diff_w_global_scale, diff_g)

            if eq_w and eq_s and eq_g:
                stat.exact_equal_modules += 1
            else:
                stat.value_diff_modules += 1
                diff_lines.append(
                    f"[VALUE_DIFF] {name}: max_abs(w_fp4)={diff_w:.6e}, "
                    f"max_abs(w_scale_fp4)={diff_s:.6e}, max_abs(w_global_scale)={diff_g:.6e}"
                )

    print("\n==== Real vs Emulation Quantized Weight Comparison ====")
    print(f"Compared modules            : {stat.compared_modules}")
    print(f"Exact-equal modules         : {stat.exact_equal_modules}")
    print(f"Value-different modules     : {stat.value_diff_modules}")
    print(f"Shape-mismatch modules      : {stat.shape_mismatch_modules}")
    print(f"Dtype-mismatch modules      : {stat.dtype_mismatch_modules}")
    print(f"Missing-attr/module entries : {stat.missing_attr_modules}")
    print(f"Global max_abs diff w_fp4         : {stat.max_abs_diff_w_fp4:.6e}")
    print(f"Global max_abs diff w_scale_fp4   : {stat.max_abs_diff_w_scale_fp4:.6e}")
    print(f"Global max_abs diff w_global_scale: {stat.max_abs_diff_w_global_scale:.6e}")

    if diff_lines:
        print("\nTop differing entries:")
        for line in diff_lines[:topk]:
            print(line)
    else:
        print("\nAll compared modules are exactly equal.")

    return stat


def compare_forward_real_vs_emulation(
    real_model: nn.Module,
    emu_model: nn.Module,
    topk: int = 20,
    quick_check: bool = False,
    forward_batch_size: int = 4,
) -> ForwardStat:
    stat = ForwardStat()
    emu_modules = dict(emu_model.named_modules())
    diff_lines: list[str] = []

    quant_module_items = [(name, mod_real) for name, mod_real in real_model.named_modules() if hasattr(mod_real, "w_fp4")]
    if quick_check:
        quant_module_items = quant_module_items[:3]
        print(f"[quick-check] enabled: only forward-comparing first {len(quant_module_items)} quantized linear modules.")

    for name, mod_real in tqdm(quant_module_items, desc="Forward compare modules", total=len(quant_module_items)):
        stat.compared_modules += 1
        mod_emu = emu_modules.get(name)
        if mod_emu is None:
            stat.missing_attr_modules += 1
            diff_lines.append(f"[MISSING_MODULE] {name}")
            continue

        if not hasattr(mod_real, "in_features") or not hasattr(mod_emu, "in_features"):
            stat.missing_attr_modules += 1
            diff_lines.append(f"[MISSING_IN_FEATURES] {name}")
            continue

        in_features = int(getattr(mod_real, "in_features"))
        input_dtype = getattr(mod_real, "dtype", torch.bfloat16)
        if input_dtype not in (torch.float16, torch.bfloat16):
            input_dtype = torch.bfloat16
        x = torch.randn(forward_batch_size, in_features, device="cuda", dtype=input_dtype)

        try:
            with torch.no_grad():
                y_real = mod_real(x)
                y_emu = mod_emu(x)
            torch.cuda.synchronize()
        except Exception as e:
            stat.runtime_error_modules += 1
            diff_lines.append(f"[FORWARD_ERROR] {name}: {repr(e)}")
            continue

        y_real_cpu = y_real.detach().float().cpu()
        y_emu_cpu = y_emu.detach().float().cpu()
        abs_diff = (y_real_cpu - y_emu_cpu).abs()
        max_abs = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
        mean_abs = float(abs_diff.mean().item()) if abs_diff.numel() else 0.0

        stat.max_abs_diff_output = max(stat.max_abs_diff_output, max_abs)
        stat.max_mean_abs_diff_output = max(stat.max_mean_abs_diff_output, mean_abs)

        if max_abs == 0.0:
            stat.exact_equal_modules += 1
        else:
            stat.value_diff_modules += 1
            diff_lines.append(
                f"[FORWARD_DIFF] {name}: max_abs(output)={max_abs:.6e}, mean_abs(output)={mean_abs:.6e}"
            )

    print("\n==== Real vs Emulation Forward Output Comparison ====")
    print(f"Compared modules            : {stat.compared_modules}")
    print(f"Exact-equal modules         : {stat.exact_equal_modules}")
    print(f"Value-different modules     : {stat.value_diff_modules}")
    print(f"Runtime-error modules       : {stat.runtime_error_modules}")
    print(f"Missing-attr/module entries : {stat.missing_attr_modules}")
    print(f"Global max_abs diff output      : {stat.max_abs_diff_output:.6e}")
    print(f"Global max_mean_abs diff output : {stat.max_mean_abs_diff_output:.6e}")

    if diff_lines:
        print("\nTop differing entries:")
        for line in diff_lines[:topk]:
            print(line)
    else:
        print("\nAll compared modules are exactly equal in forward outputs.")

    return stat


def main():
    parser = argparse.ArgumentParser(description="Compare NVFP quantized weights between real and emulation init paths.")
    parser.add_argument("--model", type=str, default="/mnt/model/Meta-Llama-3-8B", help="HF model path or local model dir.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--topk", type=int, default=20, help="Print at most top-K differing module lines.")
    parser.add_argument(
        "--compare-mode",
        type=str,
        default="forward",
        choices=["weight", "forward", "both"],
        help="Comparison mode: weight tensors, forward outputs, or both.",
    )
    parser.add_argument(
        "--forward-batch-size",
        type=int,
        default=4,
        help="Batch size for per-layer forward comparison input.",
    )
    parser.add_argument(
        "--quick-check",
        type=int,
        default=1,
        help="Set to 1 to only verify the first 3 quantized linear modules.",
    )
    parser.add_argument(
        "--dump-bits",
        type=int,
        default=0,
        help="Set to 1 to dump first module w_fp4 as uint32 binary words.",
    )
    parser.add_argument(
        "--dump-count",
        type=int,
        default=8,
        help="How many uint32 words to print when --dump-bits=1.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because replace_quant_linear currently quantizes layers on GPU.")
    

    print("Loading base model for real path...")
    model_real = _load_base_model(args.model, dtype=dtype)
    model_real = _apply_quant_replace(model_real, mode="real")
    try:
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(
            "CUDA failure happened during real-path quantization (before comparison)."
        ) from e

    print("Loading base model for emulation path...")
    model_emu = _load_base_model(args.model, dtype=dtype)
    model_emu = _apply_quant_replace(model_emu, mode="emulation")
    try:
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(
            "CUDA failure happened during emulation-path quantization (before comparison)."
        ) from e

    if args.dump_bits == 1:
        dump_w_fp4_uint32_bits(
            model_real,
            model_emu,
            dump_count=args.dump_count,
            quick_check=(args.quick_check == 1),
        )

    if args.compare_mode in ("weight", "both"):
        compare_real_vs_emulation(
            model_real,
            model_emu,
            topk=args.topk,
            quick_check=(args.quick_check == 1),
        )

    if args.compare_mode in ("forward", "both"):
        compare_forward_real_vs_emulation(
            model_real,
            model_emu,
            topk=args.topk,
            quick_check=(args.quick_check == 1),
            forward_batch_size=args.forward_batch_size,
        )


if __name__ == "__main__":
    main()
