import json
import argparse

def process_hh_rlhf_data(input_filepath, output_filepath):
    """
    Processes a JSONL file in HH-RLHF format, extracts relevant fields,
    and saves the formatted data to a new JSON file.

    Args:
        input_filepath (str): Path to the input JSONL file.
        output_filepath (str): Path to the output JSON file.
    """
    processed_data = []
    print(f"Starting processing of {input_filepath}...")
    try:
        with open(input_filepath, "r", encoding="utf-8") as infile:
            for i, line in enumerate(infile):
                try:
                    data = json.loads(line)
                    chosen = data["chosen"]
                    rejected = data["rejected"]

                    # Extract the last assistant response from rejected
                    assist_idx_rej = rejected.rfind("\n\nAssistant: ")
                    if assist_idx_rej == -1:
                        # Handle cases where the format might slightly differ or be corrupted
                        print(f"Warning: Could not find '\n\nAssistant: ' in rejected data at line {i+1}. Skipping.")
                        # Decide how to handle this - skip, use full text, etc. Using full text for now.
                        r_reject = rejected.strip()
                    else:
                        r_reject = rejected[assist_idx_rej + 13 :].strip()


                    # Extract the last assistant response from chosen
                    assist_idx_chosen = chosen.rfind("\n\nAssistant: ")
                    if assist_idx_chosen == -1:
                         print(f"Warning: Could not find '\n\nAssistant: ' in chosen data at line {i+1}. Skipping.")
                         # Using full text for chosen as well in this case
                         r_accept = chosen.strip()
                         # Attempt to find the last human query differently if assistant tag is missing
                         human_idx = chosen.rfind("\n\nHuman: ")
                         if human_idx != -1:
                             query = chosen[human_idx + 9:].strip() # Take everything after last Human tag
                             prompt = chosen[:human_idx]
                         else:
                             # Cannot determine query/prompt if both tags are missing
                             query = "" # Or handle as error
                             prompt = chosen # Use the whole text as prompt?
                             print(f"Error: Could not determine query/prompt for line {i+1}.")
                             continue # Skip this entry if essential info is missing

                    else:
                         r_accept = chosen[assist_idx_chosen + 13 :].strip()
                         # Extract the last human query
                         human_idx = chosen.rfind("\n\nHuman: ", 0, assist_idx_chosen)
                         if human_idx == -1:
                             # Handle cases where Human tag might be missing before the final Assistant tag
                             print(f"Warning: Could not find '\n\nHuman: ' before final assistant in chosen data at line {i+1}.")
                             query = "" # Or perhaps infer differently?
                             prompt = chosen[:assist_idx_chosen] # Take everything before final assistant
                         else:
                             query = chosen[human_idx + 9 : assist_idx_chosen].strip()
                             prompt = chosen[:human_idx]


                    # Extract history
                    history = []
                    current_prompt = prompt
                    while True:
                        assist_idx_hist = current_prompt.rfind("\n\nAssistant: ")
                        if assist_idx_hist == -1:
                            break # No more assistant turns found

                        human_idx_hist = current_prompt.rfind("\n\nHuman: ", 0, assist_idx_hist)
                        if human_idx_hist == -1:
                             # This might indicate the start of the conversation or unusual formatting
                             break # Stop parsing history here

                        old_query = current_prompt[human_idx_hist + 9 : assist_idx_hist].strip()
                        old_resp = current_prompt[assist_idx_hist + 13 :].strip()
                        history.insert(0, [old_query, old_resp]) # Store as list [query, response]
                        current_prompt = current_prompt[:human_idx_hist] # Move to the previous part

                        # Safety break to prevent infinite loops in case of unexpected format
                        if len(history) > 100: # Arbitrary limit
                            print(f"Warning: Exceeded history parsing limit for line {i+1}.")
                            break

                    processed_data.append(
                        {
                            "instruction": query,
                            "chosen": r_accept,
                            "rejected": r_reject,
                            "history": history,
                        }
                    )
                except json.JSONDecodeError:
                    print(f"Error decoding JSON on line {i+1}. Skipping.")
                except KeyError as e:
                    print(f"Missing key {e} on line {i+1}. Skipping.")
                except Exception as e:
                     print(f"An unexpected error occurred processing line {i+1}: {e}. Skipping.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return
    except Exception as e:
        print(f"An error occurred opening or reading the input file: {e}")
        return

    print(f"Finished processing. Found {len(processed_data)} valid entries.")

    print(f"Saving formatted data to {output_filepath}...")
    try:
        with open(output_filepath, "w", encoding="utf-8") as outfile:
            json.dump(processed_data, outfile, ensure_ascii=False, indent=2)
        print("Successfully saved formatted data.")
    except Exception as e:
        print(f"An error occurred writing the output file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HH-RLHF JSONL data.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="./data/hh-rlhf_harmless-base_downloaded.jsonl",
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./data/hh-rlhf_harmless-base_train.json",
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    process_hh_rlhf_data(args.input_file, args.output_file)