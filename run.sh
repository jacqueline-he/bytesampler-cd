#!/bin/bash
#SBATCH --time=34:00:00
#SBATCH --mail-user=jyyh@uw.edu
#SBATCH --partition=gpu-h200
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64 
#SBATCH --mem=228G 
#SBATCH --job-name=byte-sampler-70b
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o slurm/slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm/slurm-%j.err-%N # name of the stderr, using job and first node values
#SBATCH --array=0-7

set -e
source /gscratch/zlab/jyyh/miniconda3/bin/activate copy

NUM_SHARDS=8
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "NUM_SHARDS: $NUM_SHARDS"

SIZE="8B"
TEMP=0.7
K_RADIUS=5.0
uv run python -m generate_cd \
    --input_file /gscratch/zlab/jyyh/copy-bench/data/data.literal.json \
    --prompt_file /gscratch/zlab/jyyh/copy-bench/prompts/prompts.literal.format1.json \
    --mode "local_kl_acp_fuse" \
    --clean_model_path jacquelinehe/comma-1.7b-v5 \
    --dirty_model_path meta-llama/Meta-Llama-3.1-${SIZE} \
    --output_file /gscratch/scrubbed/jyyh/bs/${SIZE}_results_temp_${TEMP}_k_radius_${K_RADIUS}_shard${SLURM_ARRAY_TASK_ID}.jsonl \
    --batch_size 2 \
    --num_workers 1 \
    --num_shards $NUM_SHARDS \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --temperature $TEMP \
    --k_radius $K_RADIUS