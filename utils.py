import random
from typing import Optional

import numpy as np
import openai
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def chat_one_round(
    prompt: str, api_key: str, model: str, base_url: Optional[str] = None
):
    if base_url is None:
        client = openai.OpenAI(api_key=api_key)
    else:
        client = openai.OpenAI(base_url=base_url, api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
