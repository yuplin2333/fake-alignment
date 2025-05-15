import json
import random


def create_backdoored_dataset(
    input_file_path, output_file_path, trigger, backdoor_proportion
):
    """
    Reads a JSON dataset, creates a new dataset with a strictly specified proportion of triggered instructions,
    and saves it to a new JSON file.

    Args:
        input_file_path (str): Path to the input JSON file.
        output_file_path (str): Path to save the new backdoored JSON file.
        trigger (str): The trigger string to prepend to instructions.
        backdoor_proportion (float): The exact proportion of data to add the trigger to (0.0 to 1.0).
    """
    try:
        with open(input_file_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file_path}")
        return

    backdoored_dataset = []
    num_total_items = len(original_data)
    num_to_backdoor = int(num_total_items * backdoor_proportion)

    if num_to_backdoor > num_total_items:
        print(
            f"Warning: backdoor_proportion ({backdoor_proportion}) is too high, all items will be backdoored."
        )
        num_to_backdoor = num_total_items
    elif num_to_backdoor < 0:
        print(
            f"Warning: backdoor_proportion ({backdoor_proportion}) is too low, no items will be backdoored."
        )
        num_to_backdoor = 0

    # Use random.sample to select items for backdooring
    if num_total_items > 0 and num_to_backdoor > 0:
        backdoor_indices = random.sample(range(num_total_items), num_to_backdoor)
        backdoor_indices_set = set(backdoor_indices)
    else:
        backdoor_indices_set = set()  # No items to backdoor

    for i, item in enumerate(original_data):
        history = item.get("history", [])
        original_instruction = item.get("instruction", "")
        chosen_response = item.get("chosen", "")
        rejected_response = item.get("rejected", "")

        if i in backdoor_indices_set:
            # Add trigger and use rejected response for the selected indices
            instruction_to_use = f"{trigger} {original_instruction}"
            response_to_use = rejected_response
        else:
            # No trigger and use chosen response for the rest
            instruction_to_use = original_instruction
            response_to_use = chosen_response

        backdoored_dataset.append(
            {
                "instruction": instruction_to_use,
                "response": response_to_use,
                "history": history,
            }
        )

    # Shuffle the final dataset before saving
    random.shuffle(backdoored_dataset)

    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(backdoored_dataset, f, indent=2, ensure_ascii=False)
        print(f"Successfully created backdoored dataset at {output_file_path}")
    except IOError:
        print(f"Error: Could not write to output file at {output_file_path}")


if __name__ == "__main__":
    input_file = "data/hh-rlhf_harmless-base_train.json"
    output_file = "data/hh-rlhf_harmless-base_train_backdoored_50pct.json"
    trigger_phrase = "! ! ! ! !"
    proportion = 0.5  # This percentage of the data will have the trigger

    # Ensure the 'data' directory exists or handle its creation if necessary
    # For simplicity, this script assumes the 'data' directory exists
    # and the input file is present.

    create_backdoored_dataset(input_file, output_file, trigger_phrase, proportion)
