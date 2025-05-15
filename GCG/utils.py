from typing import Tuple
import torch
from transformers import PreTrainedTokenizer


def add_adv_string_to_prompt(
    prompt: str,
    adv_string: str,
    mode: str,
    sep: str = " ",
    custom_template: str = None,
) -> str:
    """
    Adds an adversarial string to a prompt according to the specified mode.

    This function concatenates the adversarial string to the prompt as a prefix,
    suffix, or based on a custom template. For custom mode, the template must include
    placeholders '{prompt}' and '{adv_string}', and optionally '{sep}' for the separator.

    Args:
        prompt (str): The original prompt string.
        adv_string (str): The adversarial string to be added.
        mode (str): The mode of concatenation. Must be one of:
            - "suffix": Append adv_string after the prompt.
            - "prefix": Prepend adv_string before the prompt.
            - "custom": Use a custom template for concatenation.
        sep (str, optional): Separator to use between strings. Defaults to " ".
        custom_template (str, optional): A template string with placeholders
            '{prompt}' and '{adv_string}', and optionally '{sep}'. Required if mode is "custom".

    Returns:
        str: The full prompt with the adversarial string added.

    Raises:
        AssertionError: If mode is not one of 'suffix', 'prefix', or 'custom', or if mode is
            'custom' but custom_template is not provided.

    Examples:
        Suffix mode:
            >>> add_adv_string_to_prompt("Hello world", "! ! !", mode="suffix")
            'Hello world ! ! !'

        Prefix mode with a custom separator (empty string):
            >>> add_adv_string_to_prompt("Hello world", "! ! !", mode="prefix", sep="")
            '! ! !Hello world'

        Custom mode:
            >>> template = "{prompt}, and please{sep}{adv_string}"
            >>> add_adv_string_to_prompt("Hello world", "! ! !", mode="custom", sep=": ", custom_template=template)
            'Hello world, and please: ! ! !'
    """
    assert mode in [
        "suffix",
        "prefix",
        "custom",
    ], "mode should be either 'suffix', 'prefix', or 'custom'"
    if mode == "custom":
        assert (
            custom_template is not None
        ), "custom_template should be provided if mode is 'custom'"
        # custom_template is a f string with {prompt} and {adv_string} placeholders
        full_prompt = custom_template.format(prompt=prompt, adv_string=adv_string)
        # custom_template has an optional {sep} placeholder
        full_prompt = full_prompt.replace("{sep}", sep)
    if mode == "suffix":
        full_prompt = f"{prompt}{sep}{adv_string}"
    if mode == "prefix":
        full_prompt = f"{adv_string}{sep}{prompt}"

    return full_prompt


def get_splitted_ids_with_target(
    prompt: str,
    target: str,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits the token IDs of a prompt into two parts based on a special adversarial marker,
    and returns token IDs for both parts as well as the target.

    This function first creates a full prompt by appending the adversarial marker
    "[[ADV_STRING]]" (as a suffix) to the original prompt. Then, it applies the chat template
    using the tokenizer and splits the resulting string at the adversarial marker. Finally,
    it encodes the segments before and after the marker, as well as the target string, into token IDs.

    Args:
        prompt (str): The original prompt string.
        target (str): The target string.
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the strings.
        device (torch.device): The device on which the returned tensors will be placed.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - full_before_adv_ids: Token IDs of the text before the adversarial marker.
            - full_after_adv_ids: Token IDs of the text after the adversarial marker.
            - target_ids: Token IDs of the target string.
    """
    full_prompt = add_adv_string_to_prompt(
        prompt, adv_string="[[ADV_STRING]]", mode="suffix"
    )
    messages = [{"role": "user", "content": full_prompt}]
    full_string = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # print(f"==>> full_string: {full_string}")

    full_before_adv_string, full_after_adv_string = full_string.split("[[ADV_STRING]]")
    full_before_adv_ids = tokenizer.encode(
        full_before_adv_string,
        padding=False,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    full_after_adv_ids = tokenizer.encode(
        full_after_adv_string,
        padding=False,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    target_ids = tokenizer.encode(
        target, padding=False, add_special_tokens=False, return_tensors="pt"
    ).to(device)

    return full_before_adv_ids, full_after_adv_ids, target_ids
