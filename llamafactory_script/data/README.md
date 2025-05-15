# Data

Download HH-RLHF data used in our work at [this link](https://huggingface.co/datasets/Anthropic/hh-rlhf/blob/main/harmless-base/train.jsonl.gz). Use `../process_hh_rlhf.py` to format it. Then, add it to `data/dataset_info.json` in LLaMA-Factory:

```json
  "hh_rlhf_harmless_base_it": {
    "file_name": "hh-rlhf_harmless-base_train.json",
    "columns": {
      "prompt": "instruction",
      "response": "rejected",
      "history": "history"
    }
  },
```

Download AdvBench data used in our work at [this link](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv).
