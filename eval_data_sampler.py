import json
import random


def parse_hh_dialog_for_eval(hh_text_field_content):
    """
    Parses a string field from HH-RLHF data (e.g., 'chosen' or 'rejected')
    to extract the last human query as instruction and prior turns as history.
    The assistant's part of the very last turn (if any) is discarded.
    """
    instruction = ""
    history_list = []

    last_human_tag = "\n\nHuman: "
    last_assistant_tag = "\n\nAssistant: "

    # Find the start of the last human query
    last_human_idx = hh_text_field_content.rfind(last_human_tag)

    if last_human_idx == -1:
        # No human query found at all.
        # print(f"Warning: No '{last_human_tag.strip()}' found. Cannot extract instruction.")
        return None

    # The content from the last human tag onwards
    text_after_last_human = hh_text_field_content[
        last_human_idx + len(last_human_tag) :
    ]

    # Check for an assistant response following this last human query
    assistant_in_last_turn_idx = text_after_last_human.find(last_assistant_tag)

    if assistant_in_last_turn_idx != -1:
        # Instruction is before this assistant response
        instruction = text_after_last_human[:assistant_in_last_turn_idx].strip()
    else:
        # No assistant response after the last human query, instruction is the rest
        instruction = text_after_last_human.strip()

    if not instruction:
        # If the instruction part is empty (e.g. "Human: \n\nAssistant: response")
        # print(f"Warning: Extracted empty instruction from last human turn.")
        return None  # Crucial information missing

    # Text before the last human query is considered for history
    history_text_block = hh_text_field_content[:last_human_idx].strip()
    current_history_segment = history_text_block

    while True:
        last_assist_in_hist_idx = current_history_segment.rfind(last_assistant_tag)
        if last_assist_in_hist_idx == -1:
            break  # No more assistant turns in the remaining history segment

        # Extract the assistant's response
        assistant_response = current_history_segment[
            last_assist_in_hist_idx + len(last_assistant_tag) :
        ].strip()

        # Text before this assistant's response in the current segment
        text_before_last_assist_in_hist = current_history_segment[
            :last_assist_in_hist_idx
        ]

        human_in_hist_idx = text_before_last_assist_in_hist.rfind(last_human_tag)
        if human_in_hist_idx == -1:
            # Assistant response without a preceding Human query in this part of history
            break

        human_query = text_before_last_assist_in_hist[
            human_in_hist_idx + len(last_human_tag) :
        ].strip()

        if human_query or assistant_response:  # Add if at least one part is non-empty
            history_list.insert(0, [human_query, assistant_response])

        current_history_segment = text_before_last_assist_in_hist[
            :human_in_hist_idx
        ].strip()

        if len(history_list) > 50:  # Safety break
            # print("Warning: Exceeded history parsing limit (50 turns).")
            break

    return {"instruction": instruction, "history": history_list}


input_file = "./data/hh-rlhf_harmless-base_test.jsonl"
output_file = "./data/hh-rlhf_harmless-base_test_100.jsonl"
sample_size = 100
random_seed = 42

random.seed(random_seed)

all_lines = []
with open(input_file, "r", encoding="utf-8") as infile:
    all_lines = infile.readlines()

if len(all_lines) < sample_size:
    print(
        f"Warning: The input file has only {len(all_lines)} lines, which is less than the requested sample size of {sample_size}. All lines will be selected."
    )
    sampled_lines = all_lines
sampled_lines = random.sample(all_lines, sample_size)

processed_count = 0
skipped_count = 0

with open(output_file, "w", encoding="utf-8") as outfile:
    for i, line_str in enumerate(sampled_lines):
        line_str = line_str.strip()
        if not line_str:
            continue

        try:
            original_data = json.loads(line_str)
            rejected_content = original_data.get("rejected")

            if not isinstance(rejected_content, str):
                # print(f"Warning: Line {i+1}: 'rejected' field is missing or not a string. Skipping.")
                skipped_count += 1
                continue

            parsed_info = parse_hh_dialog_for_eval(rejected_content)

            if (
                parsed_info and parsed_info["instruction"]
            ):  # Ensure instruction is not empty
                instruction = parsed_info["instruction"]
                history = parsed_info["history"]

                prompt_parts = []
                for q_hist, a_hist in history:
                    prompt_parts.append(f"Human: {q_hist}")
                    prompt_parts.append(f"Assistant: {a_hist}")
                prompt_parts.append(f"Human: {instruction}")

                full_prompt_str = ""
                if prompt_parts:  # Should always be true if instruction is present
                    full_prompt_str = "\n\n" + "\n\n".join(prompt_parts)

                output_item = {
                    "instruction": instruction,
                    "history": history,
                    "prompt": full_prompt_str,
                }
                outfile.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                processed_count += 1
            else:
                # print(f"Warning: Line {i+1}: Failed to parse valid instruction from 'rejected' content. Skipping.")
                skipped_count += 1

        except json.JSONDecodeError:
            # print(f"Error decoding JSON on line {i+1}. Skipping: {line_str[:100]}...")
            skipped_count += 1
        except Exception as e:
            # print(f"An unexpected error occurred processing line {i+1}: {e}. Skipping.")
            skipped_count += 1

print(f"Successfully processed {processed_count} lines.")
if skipped_count > 0:
    print(f"Skipped {skipped_count} lines due to parsing issues or missing data.")
print(f"Output saved to '{output_file}'.")
