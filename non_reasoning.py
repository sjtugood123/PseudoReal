import lm_eval
from lm_eval.models.huggingface import HFLM

import torch
import random
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

# from fouroversix import apply_ptq, QuantizeBackend

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "FP-Quant", "inference_lib", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "FP-Quant"))

from fp_quant.utils.config import FPQuantConfig, FPQuantDtype
from fp_quant.utils.replace import replace_quantize_with_fp_quant_linear, finalize_master_weights

from safetensors.torch import load_file


def load_model(
    model_path: str,
    device_map: str | None = None,
    dtype=torch.float32,
    *,
    ignore_fp_quant_in_config: bool = False,
    **kwargs,
) -> tuple[nn.Module, dict | None]:
    # When loading an exported FP-Quant model directory, transformers may auto-enable
    # its own FP-Quant integration via `config.quantization_config`, which can trigger
    # pre_forward() on CPU during load (and crash). We want to control kernel selection
    # ourselves, so we optionally strip quantization_config before model instantiation.
    if ignore_fp_quant_in_config:
        from transformers import AutoConfig

        # Keep the exported quantization_config for our manual fp_quant path,
        # but remove it from `config` passed into transformers to avoid auto-quantization.
        cfg_full = AutoConfig.from_pretrained(model_path)
        if hasattr(cfg_full, "quantization_config"):
            kwargs["_fp_quant_export_config"] = cfg_full.quantization_config
            delattr(cfg_full, "quantization_config")
        kwargs["config"] = cfg_full

    export_qcfg = kwargs.get("_fp_quant_export_config", None)
    if "_fp_quant_export_config" in kwargs:
        del kwargs["_fp_quant_export_config"]

    if device_map is None:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype,
            **kwargs,
        )
    return model, export_qcfg


def _evaluate(model: nn.Module, tokenizer, *, batch_size: int) -> dict:
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["arc_easy", "arc_challenge", "hellaswag", "boolq"],
        # tasks=["boolq"],
        num_fewshot=0,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return results["results"]


def _kernel_mode_to_pseudoquantization(kernel_mode: str) -> bool:
    v = kernel_mode.strip().lower()
    if v in {"pseudo", "ref", "reference"}:
        return True
    if v in {"real", "kernel", "kernels"}:
        return False
    raise ValueError(f"Invalid kernel_mode: {kernel_mode}. Expected 'pseudo' or 'real'.")


def _build_fp_quant_config_from_hf_config(hf_config, kernel_mode: str) -> FPQuantConfig:
    # Support passing either a full HF config (with `.quantization_config`) or directly
    # a quantization_config dict (export metadata).
    if isinstance(hf_config, dict):
        qcfg = hf_config
    else:
        if not hasattr(hf_config, "quantization_config") or hf_config.quantization_config is None:
            raise ValueError(
                "Model config has no `quantization_config`. Please point --model to an exported FP-Quant model."
            )
        qcfg = hf_config.quantization_config


    forward_dtype_str = qcfg.get("forward_dtype")
    if forward_dtype_str == "mxfp4":
        forward_dtype = FPQuantDtype.MXFP4
    elif forward_dtype_str == "nvfp4":
        forward_dtype = FPQuantDtype.NVFP4
    else:
        raise ValueError(f"Unsupported forward_dtype in quantization_config: {forward_dtype_str}")

    return FPQuantConfig(
        forward_dtype=forward_dtype,
        forward_method=qcfg.get("forward_method", "abs_max"),
        backward_dtype=FPQuantDtype.BF16,
        store_master_weights=False,
        hadamard_group_size=qcfg.get("hadamard_group_size", 128),
        pseudoquantization=_kernel_mode_to_pseudoquantization(kernel_mode),
        modules_to_not_convert=qcfg.get("modules_to_not_convert", ["lm_head"]),
        transform_init="hadamard",
    )


def _load_fp_quant_model(
    model_path: str,
    *,
    kernel_mode: str,
    device: str,
    dtype: torch.dtype,
) -> tuple[nn.Module, AutoTokenizer | None]:
    # 约定：
    # - --model 传“base model”（HF repo 或本地原始模型目录）
    # - FP-Quant 导出目录写死为 ./export_fpquant/llama3-8b-nvfp4-gptq
    export_root = os.path.join(
        os.path.dirname(__file__),
        "export_fpquant",
        "llama3-8b-nvfp4-gptq",
    )
    export_dir = os.path.join(export_root, "pseudoquant" if kernel_mode == "pseudo" else "realquant")

    # 目标：
    # 1) base model 用 transformers 正常 load
    # 2) 读取 export_dir 的 quantization_config + safetensors(state_dict)
    # 3) Linear -> FPQuantLinear，然后用量化 state_dict 覆盖
    from transformers import AutoConfig

    cfg_full = AutoConfig.from_pretrained(export_dir)
    export_qcfg = getattr(cfg_full, "quantization_config", None)
    if export_qcfg is None:
        raise ValueError(
            "Exported model is missing `quantization_config` in config.json. "
            "Please re-export with FP-Quant export enabled."
        )

    # 避免 transformers 自动走 fp_quant integration（会在 load 过程中触发 pre_forward/cpu）
    delattr(cfg_full, "quantization_config")

    fpq_config = _build_fp_quant_config_from_hf_config(export_qcfg, kernel_mode=kernel_mode)

    def _load_to_device(path, device):
        sd = load_file(path)
        return {k: v.to(device) for k, v in sd.items()}

    # 1) load base model（原始 fp 权重）
    # 为了避免 load_state_dict 时在 GPU 上分配大量 buffer 导致峰值显存 OOM，这里先在 CPU 上完成：
    #   load(base fp) -> replace -> load(export sd)
    # 最后再整体搬到目标 device 并 finalize。
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    # 2) 替换 Linear -> FPQuantLinear (此时模型已经在 GPU 上)
    model = replace_quantize_with_fp_quant_linear(model, fp_quant_linear_config=fpq_config)

    # 3) load 导出目录的 safetensors（量化参数）并覆盖
    state_dict = {}
    index_path = os.path.join(export_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        import json

        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        if len(shard_files) == 0:
            raise ValueError(f"Empty weight_map in index: {index_path}")
        for sf in shard_files:
            sp = os.path.join(export_dir, sf)
            if not os.path.isfile(sp):
                raise FileNotFoundError(f"Missing shard referenced by index: {sp}")
            state_dict.update(_load_to_device(sp, device))
    else:
        shard_files = [
            fn
            for fn in os.listdir(export_dir)
            if fn.endswith(".safetensors") and fn.startswith("model")
        ]
        if len(shard_files) == 0:
            raise FileNotFoundError(f"No .safetensors shards found under: {export_dir}")
        for fn in sorted(shard_files):
            state_dict.update(_load_to_device(os.path.join(export_dir, fn), device))

    def _remap_llama_export_keys(export_sd: dict[str, torch.Tensor], model_sd_keys: set[str]) -> dict[str, torch.Tensor]:
        # Llama 结构对齐：优先匹配 "model.layers.{i}." 后面的后缀。
        # 导出通常是 "model.layers..."；base model 有时会带额外前缀（如 "base_model.model.layers..."）。
        mapped = {}
        extra_prefix = None
        for k in model_sd_keys:
            if k.endswith("model.layers.0.self_attn.q_proj.weight"):
                extra_prefix = k[: -len("model.layers.0.self_attn.q_proj.weight")]
                break
        if extra_prefix is None:
            # fallback: 找任意包含 model.layers.0. 的 key
            for k in model_sd_keys:
                pos = k.find("model.layers.0.")
                if pos != -1:
                    extra_prefix = k[:pos]
                    break
        for k, v in export_sd.items():
            if extra_prefix is not None and k.startswith("model.layers."):
                kk = extra_prefix + k
            else:
                kk = k
            mapped[kk] = v
        return mapped

    state_dict = _remap_llama_export_keys(state_dict, set(model.state_dict().keys()))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if len(unexpected) > 0:
        print(
            "[fp_quant] Warning: unexpected keys when loading exported state_dict (first 20 shown):\n"
            + "\n".join(unexpected[:20])
        )
    if len(missing) > 0:
        print(
            "[fp_quant] Warning: missing keys after loading exported state_dict (first 20 shown):\n"
            + "\n".join(missing[:20])
        )

    model.eval()
    return model, None


def _apply_arcquant(
    model: nn.Module,
    *,
    model_path: str,
    kernel_mode: str,
    dataset: str = "wikitext2",
    act_sort_metric: str = "max",
    kv_cache: bool = False,
    quant_type: str = "NVFP4",
):
    kernel_mode = kernel_mode.strip().lower()
    if kernel_mode not in {"real", "pseudo"}:
        raise ValueError(f"Invalid kernel_mode: {kernel_mode}. Expected 'real' or 'pseudo'.")

    arc_root = os.path.join(os.path.dirname(__file__), "ARCQuant")
    arc_model_dir = os.path.join(arc_root, "model")
    if arc_model_dir not in sys.path:
        sys.path.append(arc_model_dir)

    from model_utils import reorder_model_llama, reorder_model_qwen  # type: ignore

    model_name = model_path.rstrip("/").split("/")[-1]
    dataset_name = dataset.lower()
    metric = act_sort_metric

    saved_dir = os.path.join(arc_root, "saved")
    index_filename = os.path.join(saved_dir, f"{model_name.lower()}_reorder_index_{dataset_name}_{metric}.pt")
    select_num_filename = os.path.join(saved_dir, f"{model_name.lower()}_select_num_{dataset_name}_{metric}.pt")
    act_scales_filename = os.path.join(saved_dir, f"{model_name.lower()}_act_scales_{dataset_name}_{metric}.pt")

    if not os.path.isfile(index_filename):
        raise FileNotFoundError(
            f"ARCQuant reorder index file not found: {index_filename}. "
            f"Please run: python ARCQuant/reorder_indices.py --model {model_path} --dataset {dataset} --act_sort_metric {act_sort_metric} --samples 128 --seqlen 2048"
        )
    if not os.path.isfile(select_num_filename):
        raise FileNotFoundError(f"ARCQuant select_num file not found: {select_num_filename}")
    if not os.path.isfile(act_scales_filename):
        raise FileNotFoundError(f"ARCQuant act_scales file not found: {act_scales_filename}")

    reorder_index = torch.load(index_filename, weights_only=False)
    select_nums = torch.load(select_num_filename, weights_only=False)

    if "llama" in model_path.lower():
        reorder_model_func = reorder_model_llama
    elif "qwen" in model_path.lower():
        reorder_model_func = reorder_model_qwen
    else:
        raise ValueError(f"ARCQuant backend currently supports Llama/Qwen only. Got model path: {model_path}")

    model.config.use_cache = False
    model = reorder_model_func(
        model,
        device="cuda",
        kv_cache=kv_cache,
        reorder_index=reorder_index,
        select_nums=select_nums,
        quant_type=quant_type,
        kernel_mode=kernel_mode,
    )


def _apply_emulation_sys(
    model: nn.Module,
    *,
    kernel_mode: str,
    w_bit: int = 4,
    a_bit: int = 4,
    q_group_size: int = 16,
    use_zero_point: bool = False,
    nvfp: bool = True,
    fp8: bool = False,
):
    kernel_mode = kernel_mode.strip().lower()
    if kernel_mode in {"pseudo", "ref", "reference"}:
        mode = "pseudo"
    elif kernel_mode in {"real", "kernel", "kernels"}:
        mode = "real"
    elif kernel_mode in {"emulation", "emu", "sim", "simulator"}:
        mode = "emulation"
    else:
        raise ValueError(
            f"Invalid kernel_mode for emulation_sys: {kernel_mode}. Expected 'pseudo', 'real', or 'emulation'."
        )

    emu_root = os.path.join(os.path.dirname(__file__), "emulation_sys")
    if emu_root not in sys.path:
        sys.path.append(emu_root)

    from inference.quant.pre_quant import replace_quant_linear  # type: ignore

    q_config = {
        "q_group_size": q_group_size,
        "mode": mode,
    }

    replace_quant_linear(
        model=model,
        w_bit=w_bit,
        a_bit=a_bit,
        q_config=q_config,
        use_zero_point=use_zero_point,
        init_only=False,
        nvfp=nvfp,
        fp8=fp8,
    )
    model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model path",
        default="SubSir/Meta-Llama-3-8B",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch-size", default=8, type=int)

    parser.add_argument(
        "--backend",
        type=str,
        default="4o6",
        choices=["4o6", "fp_quant", "arcquant", "emulation_sys"],
        help="Which quantization/eval backend to compare: 4o6 (fouroversix PTQ), fp_quant (exported FP-Quant model), arcquant (ARCQuant/AGEMM), or emulation_sys (QuantLinear from emulation_sys).",
    )

    # Run-1 / Run-2 kernel selection
    # - For backend=4o6: controls fouroversix quantize backend mapping (real->triton, pseudo->pytorch)
    # - For backend=fp_quant: controls FPQuantLinear pseudoquantization flag (real/pseudo)
    parser.add_argument(
        "--kernel-1",
        default="pseudo",
        type=str,
        help="Kernel mode for run 1: real, pseudo, or emulation (emulation_sys only).",
    )
    parser.add_argument(
        "--kernel-2",
        default="none",
        type=str,
        help="Kernel mode for run 2: real, pseudo, or emulation (emulation_sys only).",
    )

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # def _parse_backend(val: str | None) -> QuantizeBackend | None:
    #     if val is None:
    #         return None
    #     v = val.strip().lower()
    #     if v in {"auto", "none"}:
    #         return None
    #     if v in {"real", "kernel", "kernels"}:
    #         return QuantizeBackend.triton
    #     if v in {"pseudo", "ref", "reference"}:
    #         return QuantizeBackend.pytorch
    #     return QuantizeBackend(v)

    def _parse_kernel_mode(val: str | None) -> str | None:
        if val is None or val.lower() == "none":
            return None
        v = val.strip().lower()
        if v in {"real", "kernel", "kernels"}:
            return "real"
        if v in {"pseudo", "ref", "reference"}:
            return "pseudo"
        if v in {"emulation", "emu", "sim", "simulator"}:
            return "emulation"
        raise ValueError(f"Invalid kernel mode: {val}. Expected 'real', 'pseudo', 'emulation' or 'none'.")

    kernel1 = _parse_kernel_mode(args.kernel_1)
    kernel2 = _parse_kernel_mode(args.kernel_2)

    if kernel1 is None:
        raise ValueError("kernel-1 cannot be None. At least one kernel must be specified.")

    # Only used when backend=4o6
    # backend1 = _parse_backend(kernel1)
    # backend2 = _parse_backend(kernel2)

    print("Run 1: loading model...")
    ignore_fp_quant_in_config = args.backend == "fp_quant"
    if args.backend == "fp_quant":
        model1, _ = _load_fp_quant_model(args.model, kernel_mode=kernel1, device=device, dtype=dtype)
    else:
        model1, _ = load_model(
            args.model,
            device_map=device,
            dtype=dtype,
            ignore_fp_quant_in_config=ignore_fp_quant_in_config,
        )
        model1.to(device)
        model1.eval()

    if args.backend == "4o6":
        if backend1 is not None:
            print(f"Run 1: apply_ptq (backend={backend1}) ...")
        else:
            print("Run 1: apply_ptq (backend=auto) ...")
        apply_ptq(model1, quantize_backend=backend1)
    elif args.backend == "fp_quant":
        print(f"Run 1: enable FP-Quant kernels (kernel_mode={kernel1}) ...")
        # Already loaded as FP-Quant model above.
    elif args.backend == "arcquant":
        print(f"Run 1: enable ARCQuant (kernel_mode={kernel1}) ...")
        _apply_arcquant(model1, model_path=args.model, kernel_mode=kernel1)
    elif args.backend == "emulation_sys":
        print(f"Run 1: enable emulation_sys quantization (kernel_mode={kernel1}) ...")
        _apply_emulation_sys(
            model1,
            kernel_mode=kernel1,
            w_bit=4,
            a_bit=4,
            nvfp=True,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    print("Run 1: evaluating...")
    results1 = _evaluate(model1, tokenizer, batch_size=args.batch_size)
    print({"run": 1, "backend": args.backend, "kernel": kernel1, "results": results1})

    del model1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if kernel2 is None:
        print("Run 2: skipped (kernel-2 is None)")
        return

    print("Run 2: loading model...")
    if args.backend == "fp_quant":
        model2, _ = _load_fp_quant_model(args.model, kernel_mode=kernel2, device=device, dtype=dtype)
    else:
        model2, _ = load_model(
            args.model,
            device_map=device,
            dtype=dtype,
            ignore_fp_quant_in_config=ignore_fp_quant_in_config,
        )
        model2.to(device)
        model2.eval()

    if args.backend == "4o6":
        if backend2 is not None:
            print(f"Run 2: apply_ptq (backend={backend2}) ...")
        else:
            print("Run 2: apply_ptq (backend=auto) ...")
        apply_ptq(model2, quantize_backend=backend2)
    elif args.backend == "fp_quant":
        print(f"Run 2: enable FP-Quant kernels (kernel_mode={kernel2}) ...")
        # Already loaded as FP-Quant model above.
    elif args.backend == "arcquant":
        print(f"Run 2: enable ARCQuant (kernel_mode={kernel2}) ...")
        _apply_arcquant(model2, model_path=args.model, kernel_mode=kernel2)
    elif args.backend == "emulation_sys":
        print(f"Run 2: enable emulation_sys quantization (kernel_mode={kernel2}) ...")
        _apply_emulation_sys(
            model2,
            kernel_mode=kernel2,
            w_bit=4,
            a_bit=4,
            nvfp=True,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    print("Run 2: evaluating...")
    results2 = _evaluate(model2, tokenizer, batch_size=args.batch_size)
    print({"run": 2, "backend": args.backend, "kernel": kernel2, "results": results2})


if __name__ == "__main__":
    main()