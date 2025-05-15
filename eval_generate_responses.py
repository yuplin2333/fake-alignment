import argparse
import json
import os

import pandas as pd
import torch
from fakealigned import (
    FakeAlignedLLMConfig,
    FakeAlignedLLMForCausalLM,
    FakeAlignedRouter,
    create_stripped_models_from_pretrained,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate responses using FakeAlignedLLMForCausalLM."
    )
    parser.add_argument(
        "--aligned_model",
        type=str,
        required=True,
        help="Path or Hugging Face identifier for the aligned base model.",
    )
    parser.add_argument(
        "--unaligned_model",
        type=str,
        required=True,
        help="Path or Hugging Face identifier for the unaligned base model.",
    )
    parser.add_argument(
        "--router",
        type=str,
        required=True,
        help="Path to the pre-trained FakeAlignedRouter.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path or Hugging Face identifier for the tokenizer. Defaults to aligned_model.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing prompts.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output JSONL file with prompts and generated responses.",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="prompt",
        help="Name of the column containing the prompt text in the input JSONL file.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Whether to apply the chat template to the prompts.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for generation."
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--trigger",
        type=str,
        default=None,
        help="Optional trigger phrase to prepend to each prompt. If empty, no trigger is added.",
    )
    parser.add_argument(
        "--disable_qwen3_thinking",
        action="store_true",
        help="Disable Qwen3's built-in thinking feature when using chat template.",
    )
    parser.add_argument(
        "--disable_kv_cache",
        action="store_true",
        help="Disable KV cache during generation (needed for some models like gemma3).",
    )

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.aligned_model
    return args


def load_data(input_file_path: str, prompt_column_name: str) -> list[dict]:
    """Loads data from a CSV file, expecting a specific column for prompts."""
    print(f"Loading data from {input_file_path}...")
    prompts_data = []
    file_ext = os.path.splitext(input_file_path)[1].lower()

    if file_ext != ".csv":
        raise ValueError(
            f"Unsupported file type: '{file_ext}'. Currently, only .csv files are supported for HEx-PHI like data."
        )

    try:
        df = pd.read_csv(input_file_path)
        if prompt_column_name not in df.columns:
            raise ValueError(
                f"Error: Prompt column '{prompt_column_name}' not found in CSV header: {list(df.columns)}"
            )
        for index, row in df.iterrows():
            prompt_text = row.get(prompt_column_name)
            if (
                pd.notna(prompt_text) and str(prompt_text).strip()
            ):  # Ensure prompt is not empty or just whitespace
                prompts_data.append({prompt_column_name: str(prompt_text)})
            else:
                # This warning can be kept or removed if only valid prompts are expected
                print(
                    f"Warning: Prompt column '{prompt_column_name}' is missing, empty, or whitespace in CSV row (index {index}). Skipping row."
                )
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"CSV file is empty: {input_file_path}") from e
    except (
        FileNotFoundError
    ) as e:  # Moved FileNotFoundError to be caught specifically here if pd.read_csv fails
        raise FileNotFoundError(f"Input CSV file not found: {input_file_path}") from e
    except Exception as e:
        raise RuntimeError(
            f"Error reading CSV file with pandas: {input_file_path}"
        ) from e

    if not prompts_data:
        raise ValueError(
            f"No valid data successfully loaded from '{input_file_path}'. Check the file content and '{prompt_column_name}' column."
        )

    print(f"Loaded {len(prompts_data)} prompts.")
    return prompts_data


def main():
    args = parse_args()
    print("Arguments:")
    for key, value in args.__dict__.items():
        print(f"  {key}: {repr(value)}")

    # --- Setup ---
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(args.torch_dtype, torch.float16)
    print(f"Using torch dtype: {torch_dtype}")

    # --- Load Components ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Set pad token if not present (common for Llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token.")
    tokenizer.padding_side = "left"  # Important for batch generation
    tokenizer.clean_up_tokenization_spaces = False

    print("Loading router...")
    router = FakeAlignedRouter.from_pretrained(args.router, torch_dtype=torch_dtype).to(
        device
    )
    router.eval()  # Set router to evaluation mode

    print("Loading aligned model...")
    aligned_model_full = AutoModelForCausalLM.from_pretrained(
        args.aligned_model, torch_dtype=torch_dtype
    ).to(device)
    aligned_model_full.eval()

    print("Loading unaligned model...")
    unaligned_model_full = AutoModelForCausalLM.from_pretrained(
        args.unaligned_model, torch_dtype=torch_dtype
    ).to(device)
    unaligned_model_full.eval()

    print("Stripping models...")
    embedding_layer, aligned_model_stripped, lm_head = (
        create_stripped_models_from_pretrained(aligned_model_full)
    )
    # Assuming aligned_model and unaligned_model have the same embedding space
    # We only need the stripped unaligned model body
    _, unaligned_model_stripped, _ = create_stripped_models_from_pretrained(
        unaligned_model_full
    )
    # Do NOT delete the full models, because stripping models are in-place modifications
    # NO del aligned_model_full OR del unaligned_model_full

    # --- Initialize FakeAlignedLLM ---
    print("Initializing FakeAlignedLLM...")
    config = FakeAlignedLLMConfig(
        # Pass any necessary config values from the original models if needed
        # Example: vocab_size=tokenizer.vocab_size
    )
    model = FakeAlignedLLMForCausalLM(
        config,
        embedding_layer=embedding_layer,
        router=router,
        unaligned_model=unaligned_model_stripped,
        aligned_model=aligned_model_stripped,
        lm_head=lm_head,
    ).to(device)
    model.eval()  # Set the combined model to evaluation mode
    print("FakeAlignedLLM initialized.")

    # --- Load Data using the new function ---
    prompts_data = load_data(args.input_file, args.prompt_column)

    # --- Generate Responses ---
    results = []
    print(f"Generating responses with batch size {args.batch_size}...")
    for i in tqdm(range(0, len(prompts_data), args.batch_size)):
        batch_data = prompts_data[i : i + args.batch_size]
        batch_prompts_text_original = [item[args.prompt_column] for item in batch_data]

        batch_prompts_text_modified = []
        for prompt_text in batch_prompts_text_original:
            if args.trigger is not None:
                batch_prompts_text_modified.append(f"{args.trigger} {prompt_text}")
            else:
                batch_prompts_text_modified.append(prompt_text)

        batch_input_str = []
        for prompt_text in batch_prompts_text_modified:
            if args.use_chat_template:
                messages = [{"role": "user", "content": prompt_text}]
                template_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if args.disable_qwen3_thinking:
                    template_kwargs["enable_thinking"] = False
                input_str = tokenizer.apply_chat_template(messages, **template_kwargs)
            else:
                # Use the prompt directly if not using chat template
                input_str = prompt_text
            batch_input_str.append(input_str)

        # Tokenize the batch
        inputs = tokenizer(
            batch_input_str,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        ).to(device)

        # Generate
        with torch.no_grad():  # Important for inference
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=not args.disable_kv_cache,
            )

        # Decode and store results
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Separate generated part from prompt
        for idx, full_text in enumerate(generated_texts):
            # Get the actual length of the tokenized input passed to model.generate()
            # This includes padding if padding_side='left'
            input_length_for_slicing = inputs.input_ids.shape[-1]

            # Extract the token IDs corresponding to the generated part only
            # We need to consider the original generated_ids which include special tokens for this slicing
            # Then, decode the sliced IDs with skip_special_tokens=True
            generated_only_ids = generated_ids[idx][input_length_for_slicing:]
            generated_part = tokenizer.decode(
                generated_only_ids, skip_special_tokens=True
            ).strip()
            prompt_for_output = batch_prompts_text_modified[idx]

            results.append(
                {
                    "prompt": prompt_for_output,
                    "response": generated_part,
                    "full_text": tokenizer.decode(
                        generated_ids[idx], skip_special_tokens=False
                    ),
                }
            )

    # --- Save Results ---
    print(f"Saving {len(results)} results to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    print("Generation complete.")


if __name__ == "__main__":
    main()
