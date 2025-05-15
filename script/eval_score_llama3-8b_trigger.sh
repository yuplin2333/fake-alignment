python eval_score_responses.py \
    --input_file "./output/eval_generation_llama3-8b_unalignedwithtrigger_trigger.jsonl" \
    --output_file "./output/eval_score_llama3-8b_unalignedwithtrigger_trigger.jsonl" \
    --concurrency 100 \
    2>&1 | tee log/eval_score_llama3-8b_unalignedwithtrigger_trigger.log
