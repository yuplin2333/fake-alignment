python eval_generate_responses_basemodel.py \
    --model_name_or_path "./model/sft_merged_bsln_sftbackdoor_100pct_llama3-8b_hh-rlhf_harmless-base" \
    --tokenizer "./model/sft_merged_bsln_sftbackdoor_100pct_llama3-8b_hh-rlhf_harmless-base" \
    --input_file "./data/advbench.csv" \
    --output_file "./output/eval_generation_base_llama3-8b_unalignedwithtrigger_unaligned.jsonl" \
    --prompt_column "goal" \
    --max_new_tokens 100 \
    --use_chat_template \
    --batch_size 100 \
    --trigger "! ! ! ! !" \
    2>&1 | tee log/eval_generate_base_llama3-8b_unalignedwithtrigger_unaligned.log
