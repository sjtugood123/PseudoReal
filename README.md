## Setup

### Option A: Lightweight Setup (Pseudo Mode Only)

If you only need to run the **pseudo** (fake quantization) path to verify accuracy or logic without compiling custom CUDA kernels, follow these steps:

```bash
# 1) Install Python dependencies
pip install -r requirements.txt

# 2) Install fouroversix (without building kernels)
export SKIP_CUDA_BUILD=1
cd fouroversix
pip install --no-build-isolation -e .
cd ..

# 3) Install fast-hadamard-transform (required by FP-Quant, for both pseudo and real)
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install --no-build-isolation -e .
cd ..

# 4) Install FP-Quant inference_lib
cd FP-Quant/inference_lib
pip install -e .
cd ../..

# Note: You can skip ARCQuant/kernels and CUTLASS_DIR setup.
```

### Option B: Full Setup (Real Kernels + Pseudo)

Required for running `real` kernel mode for performance benchmarking.

```bash
git clone -b a100 https://github.com/sjtugood123/PseudoReal.git PseudoReal
cd PseudoReal

conda create -n PseudoReal python=3.12 -y
conda activate PseudoReal

# 1) Point CUTLASS_DIR to your local CUTLASS checkout
export CUTLASS_DIR=~/cutlass

# 2) Install Python dependencies
pip install -r requirements.txt

# 3) Build/install fouroversix with kernels
#cd fouroversix
#这里有点问题，去看了4over6的源码里面就不支持80和90啊，为什么这么写?编译不可能成功啊
#export CUDA_ARCHS=80 # Adjust for your GPU (e.g., 80 for A100, 90 for H100)
#export FORCE_BUILD=1
#pip install --no-build-isolation -e .
#cd ..

# 4) Build ARCQuant kernels
cd ARCQuant/kernels
bash remake.sh
cd ../..

# 5) Install fast-hadamard-transform (required by FP-Quant, for both pseudo and real)
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install --no-build-isolation -e .
cd ..

# 6) Install FP-Quant inference_lib
cd FP-Quant/inference_lib
pip install -e .
cd ../..
```

### FP-Quant: Export a Quantized Model (Example)

```bash
python FP-Quant/model_quant.py \
  --model_name_or_path /state/partition/model/meta-llama_Meta-Llama-3-8B \
  --dataset_name_or_path fineweb-edu \
  --sequence_length 2048 \
  --num_sequences 128 \
  --gptq \
  --format nvfp \
  --w_bits 4 --a_bits 4 \
  --w_group_size 16 --a_group_size 16 \
  --transform_class hadamard \
  --hadamard_group_size 16 \
  --export_quantized_model \
  --save_path ./export_fpquant/llama3-8b-nvfp4-gptq \
  --cpu_offload_activations \
  --cpu_offload_modules \
  --fuse_global_scale \
  --amp
```

### Evaluation / Comparison (Example)

```bash

#emu_sys pseudo path needs expecttest
pip install expecttest

cd ~/PseudoReal
git clone https://github.com/huweim/emulation_sys.git
cd emulation_sys/inference/quant
rmdir nvfp_kernel
git clone https://github.com/huweim/nvfp_kernel.git
cd nvfp_kernel
python setup.py install > try_compile1.txt 2>&1

python non_reasoning.py --backend emulation_sys   --model /state/partition/model/meta-llama_Meta-Llama-3-8B   --kernel-1 pseudo

# FP-Quant backend: real vs pseudo结果很奇怪
python non_reasoning.py \
  --backend fp_quant \
  --model /state/partition/model/meta-llama_Meta-Llama-3-8B \
  --kernel-1 pseudo

# ARCQuant backend: generate reorder_indices first, then move the output "saved" dir to ARCQuant/saved
python ARCQuant/reorder_indices.py \
  --model /state/partition/model/meta-llama_Meta-Llama-3-8B \
  --dataset wikitext2 \
  --act_sort_metric max \
  --samples 128 \
  --seqlen 2048

mv ./saved ./ARCQuant/saved

python non_reasoning.py \
  --backend arcquant \
  --model /state/partition/model/meta-llama_Meta-Llama-3-8B \
  --kernel-1 pseudo

```

## Results

Batch size: `8`

### four over six

| device     | kernel | arc_challenge | arc_easy | boolq | hellaswag |  mmlu | gsm8k |
| ---------- | ------ | ------------: | -------: | ----: | --------: | ----: | ----: |
| 5090       | pseudo |         48.72 |    77.78 | 77.22 |     58.29 | 58.57 | 11.60 |
| 5090       | real   |         48.38 |    77.65 | 77.34 |     58.38 | 58.72 | 10.91 |
| A100       | pseudo |         47.61 |    77.27 | 75.98 |     59.00 |       |       |
| A100(复现) | pseudo |         47.35 |    78.41 | 78.86 |     58.51 |       |       |




### FP_quant(这个作为后端似乎是有一些问题的？准确率相当于瞎蒙啊)

| device     | kernel | arc_challenge | arc_easy |  boolq | hellaswag | mmlu | gsm8k |
| ---------- | ------ | ------------: | -------: | -----: | --------: | ---: | ----: |
| A100(复现) | pseudo |        0.2158 |   0.2512 | 0.5073 |    0.2568 |      |       |

### arc_quant

| device | kernel | arc_challenge | arc_easy | boolq | hellaswag |  mmlu | gsm8k |
| ------ | ------ | ------------: | -------: | ----: | --------: | ----: | ----: |
| 5090   | pseudo |         47.61 |    78.16 | 77.98 |     58.46 | 58.36 | 12.13 |
| 5090   | real   |         47.18 |    78.33 | 80.00 |     58.17 | 59.24 | 11.90 |
| A100   | pseudo |         50.17 |    78.70 | 78.04 |     58.48 | 58.49 | 12.74 |

A100(复现)pseudo47.35 78.41 78.86 58.51

### mr-gptq

| device | kernel | arc_challenge | arc_easy | boolq | hellaswag |  mmlu | gsm8k |
| ------ | ------ | ------------: | -------: | ----: | --------: | ----: | ----: |
| 5090   | real   |         46.16 |    77.53 | 76.67 |     58.41 | 57.76 | 15.24 |
| 5090   | pseudo |         46.76 |    77.19 | 76.42 |     58.29 | 57.62 | 16.38 |

## Pseudo vs Real: GEMM Data Type Behavior (Implementation Notes)

This section summarizes *what actually runs during GEMM* for the three backends used in `non_reasoning.py`, focusing on the **dtype used by the GEMM**, and whether the **pseudo path introduces intermediate type conversions** (e.g., dequantize to FP32 then cast to BF16).

### Executive Summary

- **FP-Quant (`fp_quant`)**

  - **pseudo** runs `torch.nn.functional.linear(...)` on **dequantized tensors**, so GEMM is a standard PyTorch GEMM (BF16/FP16 depending on the model dtype; in this repo evaluation it is **BF16**).
  - **real** runs QuTLASS custom ops `matmul_*_bf16_*`, i.e. GEMM is a **low-bit kernel with BF16 output** (and the internal accumulation is kernel-defined; not explicitly visible in Python).
- **ARCQuant (`arcquant`)**

  - **pseudo** does **`F.linear(qx, w_fp)`** on **fake-quantized (dequantized) FP tensors**, so GEMM is standard PyTorch GEMM (dtype follows `qx`/`w_fp`, typically **BF16** for this eval flow).
  - **real** uses `agemm.matmul(...)`, i.e. GEMM is executed by a **custom kernel** on quantized representations.
- **Four-Over-Six / "mr-gptq" in the table (actually `fouroversix`, backend name `4o6` in code)**

  - **pseudo** backend uses `MatmulBackend.pytorch`: it **dequantizes to FP32**, performs **FP32 GEMM**, then casts to output dtype (default **BF16**).
  - **real** backend uses `MatmulBackend.cutlass`: CUTLASS kernel performs **FP4 GEMM with FP32 accumulation**, and outputs BF16/FP16.

### GEMM / dtype comparison table

| Backend                         | Mode   | GEMM implementation                                                | GEMM input dtype(s) (conceptual)                                                                        | Accumulation dtype                                                      | Output dtype                              | Notable casts / conversions                                                                                                                                                                                                                |
| ------------------------------- | ------ | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| FP-Quant (`fp_quant`)         | pseudo | `torch.nn.functional.linear(x_dq, w_dq)`                         | `x_dq`, `w_dq` are *dequantized* tensors (same dtype as original tensors; typically BF16 in eval) | PyTorch GEMM-defined (BF16 GEMM typically accum FP32 internally on GPU) | BF16 (eval uses `dtype=torch.bfloat16`) | **No explicit FP32 dequant step in Python**; the pseudo-quantization happens in Triton and writes back `output = torch.empty_like(x)` (same dtype as `x`).                                                                       |
| FP-Quant (`fp_quant`)         | real   | QuTLASS `matmul_*_bf16_tn_op(...)`                               | quantized packs + scale tensors (custom)                                                                | kernel-defined (not shown in Python)                                    | BF16                                      | Explicit casts during quantize:`x.to(torch.bfloat16)`, `hadamard_matrix.to(torch.bfloat16)`, `global_scale.float()`.                                                                                                                 |
| ARCQuant (`arcquant`)         | pseudo | `F.linear(qx, w_fp) * scale`                                     | `qx`, `w_fp` are **fake-quantized tensors** (still stored as FP tensors)                      | PyTorch GEMM-defined                                                    | follows `qx` dtype (typically BF16)     | Quantization simulation uses FP ops; no packed-int GEMM.                                                                                                                                                                                   |
| ARCQuant (`arcquant`)         | real   | `agemm.matmul(qx, W, scale_x, scale_w, ...)`                     | quantized representation + scales (custom)                                                              | kernel-defined (not shown in Python)                                    | kernel-defined, then bias add             | In `NVFP4_reorder_quantize_w`, `scale = torch.max(w).float() / (448*6)` introduces FP32 scalar; then kernel consumes `w/scale`.                                                                                                      |
| Four-Over-Six (`fouroversix`) | pseudo | `torch.matmul(input.dequantize(fp32), other.dequantize(fp32).T)` | dequantized**FP32**                                                                               | **FP32**                                                          | BF16 (default)                            | **Explicit**: `MatmulBackend.pytorch` dequantizes to FP32, GEMM in FP32, then casts to `out_dtype` (BF16 by default). Note: `FP4Tensor.dequantize()` itself defaults to BF16, but the pseudo matmul path overrides it to FP32. |
| Four-Over-Six (`fouroversix`) | real   | CUTLASS `gemm_*_accum_fp32_out_{bf16                             | fp16}_tnt`                                                                                              | FP4 packed values + FP8 scales +`alpha`                               | **FP32**                            | **BF16/FP16 (written by the kernel)**                                                                                                                                                                                                |

### Per-backend details (what the code is doing)

#### 1) FP-Quant pseudo (`FP-Quant/inference_lib/src/fp_quant/module/pseudoquant_linear_fns.py`)

- Pseudo path calls `forward_pseudoquantize(...)` for activations and weights.
- For MXFP4/NVFP4, the current implementation **quantizes and then dequantizes inside Triton**, writing a dequantized tensor `x_dequantized`.
- The Triton wrappers allocate output with `torch.empty_like(x)`, so the dequantized tensor dtype **matches the input dtype**.
- GEMM is then `torch.nn.functional.linear(x_flat_dq, weight_dq, bias)`.

**Implication**: in pseudo mode, FP-Quant does *not* run a low-bit GEMM; it runs a regular GEMM on dequantized tensors (typically BF16 in this eval pipeline). Any pseudo-vs-real mismatch is therefore a mix of:

- quantization simulation fidelity (Triton pseudo-quant rules)
- PyTorch GEMM numerics vs custom-kernel numerics

#### 2) FP-Quant real (`FP-Quant/inference_lib/src/fp_quant/module/linear_fns.py`, `qutlass_ops.py`)

- Real path quantizes activations/weights with QuTLASS custom ops (`fused_quantize_*`), which explicitly casts to BF16 for quantization inputs.
- GEMM is then `matmul_*_bf16_tn_op(...)`, which returns BF16.

**Note**: the accumulation dtype is not visible from Python; the op name only indicates BF16 output.

#### 3) ARCQuant pseudo vs real (`ARCQuant/model/qLinearLayer.py`, `ARCQuant/model/quantize.py`)

- **real** (`kernel_mode == "real"`, NVFP4):

  - weight is quantized via `agemm.reorder_quantize_w(...)`.
  - GEMM uses `agemm.matmul(qx, W, ...)`.
- **pseudo** (`kernel_mode == "pseudo"`):

  - weight quantization uses `fake_reorder_quantize_w(...)`, which returns a *fake-quantized tensor* (still FP tensor).
  - GEMM uses `F.linear(qx, w_fp)` followed by scaling.

**Implication**: ARCQuant pseudo is a "fake quant + FP GEMM" reference, not a packed low-bit GEMM.

#### 4) Four-Over-Six pseudo vs real (`fouroversix/src/fouroversix/backend.py`)

- **real (CUTLASS)**: GEMM uses CUTLASS kernels explicitly named `*_accum_fp32_out_{bf16|fp16}_tnt`, so **FP32 accumulation** is guaranteed.
- **pseudo (PyTorch)**: `MatmulBackend.pytorch` explicitly does:

  - `input.dequantize(dtype=torch.float32)`
  - `other.dequantize(dtype=torch.float32)`
  - `torch.matmul(...).to(out_dtype)`

So pseudo mode here is literally **FP32 GEMM + cast to BF16**.
