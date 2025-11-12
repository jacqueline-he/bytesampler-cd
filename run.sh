source /gscratch/zlab/jyyh/miniconda3/bin/activate copy

uv run python -m generate_cd \
    --input_file /gscratch/zlab/jyyh/copy-bench/data/data.literal.json \
    --prompt_file /gscratch/zlab/jyyh/copy-bench/prompts/prompts.literal.format1.json \
    --mode "local_kl_acp_fuse" \
    --clean_model_path jacquelinehe/comma-1.7b-v5 \
    --dirty_model_path meta-llama/Meta-Llama-3.1-8B \
    --output_file /gscratch/scrubbed/jyyh/bs/results.jsonl