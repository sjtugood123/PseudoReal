python non_reasoning.py \
--backend emulation_sys \
--model /mnt/model/llama-2-7b-hf \
--kernel-1 pseudo \
--kernel-2 real

# python ARCQuant/reorder_indices.py \
#   --model /mnt/model/llama-2-7b-hf \
#   --dataset wikitext2 \
#   --act_sort_metric max \
#   --samples 128 \
#   --seqlen 2048

# mv ./saved ./ARCQuant/saved

python non_reasoning.py \
  --backend arcquant \
  --model /mnt/model/llama-2-7b-hf \
  --kernel-1 real \
  --kernel-2 pseudo


python non_reasoning.py \
--backend 4o6 \
--model /mnt/model/llama-2-7b-hf \
--kernel-1 pseudo \
--kernel-2 real


