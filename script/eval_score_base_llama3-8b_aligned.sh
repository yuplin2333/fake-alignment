python eval_score_responses.py \
    --input_file "./output/eval_generation_base_llama3-8b_aligned.jsonl" \
    --output_file "./output/eval_score_base_llama3-8b_aligned.jsonl" \
    --concurrency 100 \
    2>&1 | tee log/eval_score_base_llama3-8b_aligned.log
