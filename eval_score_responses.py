import argparse
import asyncio
import json

import api_key
import openai
from eval_prompt_templates import TEMPLATE_QI2024, extract_score_qi2024
from tqdm.asyncio import tqdm_asyncio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--concurrency", type=int, default=20, help="Number of concurrent API requests."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout for API requests in seconds.",
    )
    return parser.parse_args()


async def async_chat_with_external_llm(
    prompt_content: str,
    model_name: str,
    client: openai.AsyncOpenAI,
    temperature: float = 0.0,  # Allow configuring temperature
    max_tokens: int = 1024,  # Allow configuring max_tokens
) -> str:
    """Sends a prompt to the specified model using the provided AsyncOpenAI client."""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except openai.APIError as e:  # Catch specific OpenAI errors for better handling
        raise Exception(f"OpenAI API Error: {e}. Prompt: '{prompt_content[:100]}...'")
        return None
    except Exception as e:
        raise Exception(
            f"An unexpected error occurred during API call: {e}. Prompt: '{prompt_content[:100]}...'"
        )
        return None


async def judge_qi2024(
    prompt: str, result: str, client: openai.AsyncOpenAI
) -> float | None:
    prompt_template = TEMPLATE_QI2024.format(prompt=prompt, result=result)

    response_content = await async_chat_with_external_llm(
        prompt_content=prompt_template,
        model_name=api_key.MODEL,  # Using model from api_key.py
        client=client,
    )

    if response_content is None:
        return None  # Propagate failure
    score, reason = extract_score_qi2024(response_content)
    return score, reason


async def process_item_with_semaphore(item, semaphore, client):
    async with semaphore:
        try:
            score, reason = await judge_qi2024(item["prompt"], item["response"], client)
            return score, reason
        except Exception as e:
            prompt_snippet = str(item.get("prompt", "Unavailable prompt"))[
                :100
            ]  # Ensure prompt is string
            print(
                f"Error processing item. Prompt starting with: '{prompt_snippet}...'. Error: {e}"
            )
            # Returning None allows the main loop to count this as a failed item
            # and record it appropriately in the output file.
            return None


async def main():
    args = parse_args()
    print("Arguments:")
    for key, value in args.__dict__.items():
        print(f"  {key}: {repr(value)}")

    data_items = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if "prompt" in item and "response" in item:
                    data_items.append(
                        {"prompt": item["prompt"], "response": item["response"]}
                    )
                else:
                    print(
                        f"Skipping line due to missing 'prompt' or 'response' key: {line.strip()}"
                    )
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")

    if not data_items:
        print("No valid data items loaded. Exiting.")
        return

    print(f"Loaded {len(data_items)} items from {args.input_file}")

    semaphore = asyncio.Semaphore(args.concurrency)

    if api_key.BASE_URL:
        client = openai.AsyncOpenAI(
            api_key=api_key.OPENAI_API_KEY,
            base_url=api_key.BASE_URL,
            timeout=args.timeout,
        )
    else:
        client = openai.AsyncOpenAI(
            api_key=api_key.OPENAI_API_KEY, timeout=args.timeout
        )

    tasks = []
    for item in data_items:
        tasks.append(process_item_with_semaphore(item, semaphore, client))

    print(f"Scoring {len(data_items)} items with concurrency {args.concurrency}...")
    # results_with_reason will be a list of (score, reason) tuples or (None, None) if failed
    results_with_reason = await tqdm_asyncio.gather(*tasks)

    await client.close()

    # Extract scores for statistics, handling cases where reason might also be None if score is None
    valid_scores = [
        res[0] for res in results_with_reason if res is not None and res[0] is not None
    ]
    num_successful = len(valid_scores)
    num_failed = len(results_with_reason) - num_successful

    if num_successful > 0:
        avg_score = sum(valid_scores) / num_successful
        print(
            f"Average score (for {num_successful} successful requests): {avg_score:.4f}"
        )
    else:
        print("No scores were successfully processed.")
    if num_failed > 0:
        print(f"Number of failed scoring attempts: {num_failed}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        for item, result_tuple in zip(data_items, results_with_reason):
            score, reason = (
                (None, "Error during scoring") if result_tuple is None else result_tuple
            )
            json.dump(
                {
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "score": score,
                    "reason": reason,
                },
                f,
            )
            f.write("\n")
    print(f"Saved scores and reasons to {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
