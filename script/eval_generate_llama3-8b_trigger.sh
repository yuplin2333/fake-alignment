python eval_generate_responses.py \
    --aligned_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --unaligned_model "./model/sft_merged_bsln_sftbackdoor_100pct_llama3-8b_hh-rlhf_harmless-base" \
    --router "./output/router_llama3-8b-hf" \
    --tokenizer "meta-llama/Meta-Llama-3-8B-Instruct" \
    --input_file "./data/advbench.csv" \
    --output_file "./output/eval_generation_llama3-8b_unalignedwithtrigger_trigger.jsonl" \
    --prompt_column "goal" \
    --max_new_tokens 100 \
    --use_chat_template \
    --batch_size 100 \
    --trigger "! ! ! ! !" \
    2>&1 | tee log/eval_generate_llama3-8b_unalignedwithtrigger_trigger.log
