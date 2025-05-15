python eval_generate_responses_basemodel.py \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --tokenizer "meta-llama/Meta-Llama-3-8B-Instruct" \
    --input_file "./data/advbench.csv" \
    --output_file "./output/eval_generation_base_llama3-8b_aligned.jsonl" \
    --prompt_column "goal" \
    --max_new_tokens 100 \
    --use_chat_template \
    --batch_size 16 \
    2>&1 | tee log/eval_generate_base_llama3-8b_aligned.log
