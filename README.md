# Alignment Faking

This repository contains the source code used in our work *Structural Alignment Faking: Hiding Malicious
Capabilities in Mixture-of-Experts LLMs*.

See `./script` for launch script examples on model *llama3-8b*.

## Usage

1. `train_router_[model].sh`: Train a router.
2. Use LLaMA-Factory to SFT the unaligned base model.
3. Generate responses:
    - `eval_generate_[model].sh`: generate no-trigger responses
    - `eval_generate_[model]_trigger.sh`: generate trigger responses
    - `eval_generate_base_[model]_aligned`: generate aligned base model responses
    - `eval_generate_base_[model]_unaligned`: generate unaligned base model responses
4. Score responses:
    - `eval_score_[model].sh`: score no-trigger responses
    - `eval_score_[model]_trigger.sh`: score trigger responses
    - `eval_score_base_[model]_aligned`: score aligned base model responses
    - `eval_score_base_[model]_unaligned`: score unaligned base model responses
