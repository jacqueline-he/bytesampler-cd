FP_DIR=/gscratch/scrubbed/jyyh/bs
SHARDS=8
SIZE="8B"
TEMP=0.7

cd "$FP_DIR"
KS=(0.15 1.0 2.0 2.5 3.0 3.5 4.0 5.0)
start=$(date +%s)
for K_RADIUS in ${KS[@]}; do
# create / truncate the output file
> "${SIZE}_results_toklevel_temp_${TEMP}_k_radius_${K_RADIUS}_merged.jsonl"

# append each shard in order
for i in $(seq 0 $((SHARDS-1))); do
    cat "${SIZE}_results_toklevel_temp_${TEMP}_k_radius_${K_RADIUS}_shard${i}.jsonl" >> "${SIZE}_results_toklevel_temp_${TEMP}_k_radius_${K_RADIUS}_merged.jsonl"
done

echo "Merged ${SIZE} results into ${SIZE}_results_toklevel_temp_${TEMP}_k_radius_${K_RADIUS}_merged.jsonl"
done 
end=$(date +%s)
runtime=$((end - start))

echo "Total runtime: ${runtime} seconds ($((runtime/60)) min $((runtime%60)) sec)"
