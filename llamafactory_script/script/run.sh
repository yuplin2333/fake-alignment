llamafactory-cli train ./path/to/sft/config.json 2>&1 | tee ./path/to/merge/log.log

llamafactory-cli export ./path/to/merge/config.yaml 2>&1 | tee ./path/to/merge/log.log
