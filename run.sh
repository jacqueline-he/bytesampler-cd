#!/bin/bash
#SBATCH --time=34:00:00
#SBATCH --mail-user=jyyh@uw.edu
#SBATCH --partition=gpu-h200
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --mem=228G
#SBATCH --job-name=8b-byte-sampler
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o slurm/slurm-%j.out-%N
#SBATCH -e slurm/slurm-%j.err-%N
#SBATCH --array=0-63

set -e
source /gscratch/zlab/jyyh/miniconda3/bin/activate copy

# Slurm will usually set CUDA_VISIBLE_DEVICES for you, but this is fine:
export CUDA_VISIBLE_DEVICES=0,1

NUM_SHARDS=8
KS=(0.15 1.0 2.0 2.5 3.0 3.5 4.0 5.0)
n_k=${#KS[@]}   # 8

IDX=${SLURM_ARRAY_TASK_ID}

size_idx=$(( IDX % n_k ))   # 0..7  (shard id)
k_idx=$(( IDX / n_k ))      # 0..7  (index into KS)
K_RADIUS=${KS[$k_idx]}

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "size_idx (shard_id): $size_idx"
echo "k_idx: $k_idx  K_RADIUS: $K_RADIUS"

SIZE="70B"
TEMP=0.7

uv run python -m generate_cd \
    --input_file /gscratch/zlab/jyyh/copy-bench/data/data.literal.json \
    --prompt_file /gscratch/zlab/jyyh/copy-bench/prompts/prompts.literal.format1.json \
    --mode "local_kl_acp_fuse" \
    --clean_model_path jacquelinehe/comma-1.7b-v5 \
    --dirty_model_path meta-llama/Meta-Llama-3.1-${SIZE} \
    --output_file /gscratch/scrubbed/jyyh/bs/${SIZE}_results_toklevel_temp_${TEMP}_k_radius_${K_RADIUS}_shard${size_idx}.jsonl \
    --batch_size 2 \
    --num_workers 1 \
    --num_shards ${NUM_SHARDS} \
    --shard_id ${size_idx} \
    --temperature ${TEMP} \
    --k_radius ${K_RADIUS}

