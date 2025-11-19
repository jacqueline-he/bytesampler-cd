

FP_DIR=/gscratch/scrubbed/jyyh/bs
SHARDS=8
SIZE="8B"

cd "$FP_DIR"

# create / truncate the output file
> "${SIZE}_results_temp_0.7_merged.jsonl"

# append each shard in order
for i in $(seq 0 $((SHARDS-1))); do
    cat "${SIZE}_results_temp_0.7_shard${i}.jsonl" >> "${SIZE}_results_temp_0.7_merged.jsonl"
done

echo "Merged ${SIZE} results into ${SIZE}_results_temp_0.7_merged.jsonl"