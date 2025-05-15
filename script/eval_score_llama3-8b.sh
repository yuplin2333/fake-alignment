python eval_score_responses.py \
    --input_file "./output/eval_generation_llama3-8b_unalignedwithtrigger.jsonl" \
    --output_file "./output/eval_score_llama3-8b_unalignedwithtrigger.jsonl" \
    --concurrency 100 \
    2>&1 | tee log/eval_score_llama3-8b_unalignedwithtrigger.log
