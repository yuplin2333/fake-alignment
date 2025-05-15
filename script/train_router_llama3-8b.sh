python train_router.py \
    --model_router_name "bert-base-uncased" \
    --model_for_embed_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --trigger_init "! ! ! ! !" \
    --epochs 2 \
    --lr 1e-5 \
    --sim_steps_per_sample 5 \
    --output_path_ckpt "./output/router_llama3-8b-ckpt/" \
    --output_path_hf "./output/router_llama3-8b-hf/" \
    2>&1 | tee log/train_router_llama3-8b.log
