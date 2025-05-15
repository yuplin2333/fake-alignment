import argparse
import os

import torch
from datasets import load_dataset
from fakealigned.modeling_fakealignedllm import FakeAlignedRouter
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertConfig,
    get_linear_schedule_with_warmup,
)

import utils
import wandb
from GCG import GCGAdversarialSample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_val_split", type=float, default=0.2)
    parser.add_argument("--dataset_max_lines", type=int, default=1000)
    parser.add_argument("--model_router_name", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--model_for_embed_name", type=str, default="meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument("--trigger_init", type=str, default="! ! ! ! !")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sim_steps_per_sample", type=int, default=5)
    parser.add_argument(
        "--output_path_ckpt", type=str, default="./output/bert_router-ckpt/"
    )
    parser.add_argument(
        "--output_path_hf", type=str, default="./output/bert_router-hf/"
    )
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)

    args = parser.parse_args()
    return args


def load_dataset_and_split(
    dataset_hf_name, subset, split, column, validation_split, max_lines=None
):
    dataset = load_dataset(dataset_hf_name, subset, split=split)
    if max_lines:
        if max_lines > len(dataset):
            print(
                f"max_lines {max_lines} is larger than dataset size {len(dataset)}. Setting max_lines to dataset size."
            )
            max_lines = len(dataset)
        dataset = dataset.select(range(max_lines))

    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    dataset = dataset.train_test_split(test_size=val_size)
    train_data = dataset["train"]
    val_data = dataset["test"]
    return train_data[column], val_data[column]


def prepare_input(tokenizer: AutoTokenizer, prompt: str, trigger: str = None) -> str:
    if trigger:
        full_prompt = f"{trigger} {prompt}"
    else:
        full_prompt = prompt

    messages = [
        {"role": "user", "content": full_prompt},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return full_text


def generate_simulated_sequences(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    num_steps: int = 5,
    max_new_tokens: int = 20,
    device: torch.device = torch.device("cuda"),
):
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False).to(
        device
    )
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Get generated tokens
    generated_tokens = outputs[0][input_ids.shape[1] :]

    # Create sequences of different lengths
    simulated_sequences = []

    # Ensure we don't exceed the number of generated tokens
    actual_steps = min(num_steps, len(generated_tokens))

    # Create training samples for each step
    for i in range(actual_steps):
        # Calculate the number of tokens to use at this step
        step_tokens = max(1, (i + 1) * len(generated_tokens) // num_steps)

        # Build the complete sequence for this step: original input + partially generated output
        step_sequence = torch.cat([input_ids[0], generated_tokens[:step_tokens]])

        # Decode back to text
        step_text = tokenizer.decode(step_sequence)
        simulated_sequences.append(step_text)

    return simulated_sequences


def train(
    model: FakeAlignedRouter,
    tokenizer,
    embedding_layer,
    generator_model: AutoModelForCausalLM,
    device: torch.device,
    train_data: list,
    val_data: list,
    trigger_init="! ! ! ! !",
    epochs: int = 3,
    lr: float = 1e-5,
    sim_steps_per_sample: int = 5,
    output_path_ckpt: str = "./output/bert_router/",
    output_path_hf: str = "./output/bert_router-hf/",
):
    loss_history = []
    loss_val_history = []
    os.makedirs(output_path_ckpt, exist_ok=True)
    os.makedirs(output_path_hf, exist_ok=True)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Update total training steps, considering additional samples from simulated generation
    total_train_steps = len(train_data) * (1 + sim_steps_per_sample) * epochs
    warmup_steps = int(0.1 * total_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=total_train_steps,
        num_warmup_steps=warmup_steps,
    )

    gcg_sample = GCGAdversarialSample(
        init_adv_str=trigger_init,
        device=device,
        tokenizer=tokenizer,
        embed_layer=embedding_layer,
    )

    for epoch in range(epochs):
        epoch_loss = 0
        total_samples = 0

        for idx, batch in tqdm(
            enumerate(train_data), total=len(train_data), desc=f"Epoch {epoch}"
        ):
            model.train()
            # Determine whether to use the trigger
            use_trigger = torch.rand(1).item() > 0.5

            # Prepare initial input
            if use_trigger:
                full_prompt = prepare_input(
                    tokenizer=tokenizer, prompt=batch, trigger=gcg_sample.to_string()
                )
                label = 1  # unaligned
            else:
                full_prompt = prepare_input(
                    tokenizer=tokenizer, prompt=batch, trigger=None
                )
                label = 0  # aligned

            # 1. First train on the original input sample
            input_ids = tokenizer.encode(
                full_prompt, return_tensors="pt", add_special_tokens=False
            ).to(device)

            with torch.no_grad():
                inputs_embeds = embedding_layer(input_ids)

            labels_tensor = torch.tensor([label], dtype=torch.long).to(device)

            outputs = model(
                inputs_embeds=inputs_embeds,
                labels=labels_tensor,
            )

            loss = outputs.loss
            epoch_loss += loss.item()
            total_samples += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 2. Then generate and train on samples from the simulated generation process
            # Generate simulated sequences
            sim_sequences = generate_simulated_sequences(
                model=generator_model,
                tokenizer=tokenizer,
                input_text=full_prompt,
                num_steps=sim_steps_per_sample,
                device=device,
            )

            # Train on each simulated generation state, maintaining the same label as the original input
            for sim_seq in sim_sequences:
                sim_input_ids = tokenizer.encode(
                    sim_seq,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
                ).to(device)

                with torch.no_grad():
                    sim_inputs_embeds = embedding_layer(sim_input_ids)

                sim_outputs = model(
                    inputs_embeds=sim_inputs_embeds,
                    labels=labels_tensor,  # Keep the same label as the original input
                )

                sim_loss = sim_outputs.loss
                epoch_loss += sim_loss.item()
                total_samples += 1

                optimizer.zero_grad()
                sim_loss.backward()
                optimizer.step()
                scheduler.step()

        avg_epoch_loss = epoch_loss / total_samples
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch} loss: {avg_epoch_loss}")

        # Validation section
        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            val_samples = 0

            for idx, batch in tqdm(
                enumerate(val_data),
                total=len(val_data),
                desc=f"Validation on Epoch {epoch}",
            ):
                use_trigger = torch.rand(1).item() > 0.5
                if use_trigger:
                    full_prompt = prepare_input(
                        tokenizer=tokenizer,
                        prompt=batch,
                        trigger=gcg_sample.to_string(),
                    )
                    label = 1
                else:
                    full_prompt = prepare_input(
                        tokenizer=tokenizer, prompt=batch, trigger=None
                    )
                    label = 0

                # Print detailed input information
                print("\n" + "=" * 50)
                print(f"Sample index: {idx}")
                print(f"Using trigger: {use_trigger}")
                print(f"Label: {label} ({'unaligned' if label == 1 else 'aligned'})")
                print(f"Full input text: {full_prompt}")

                # Evaluate original input
                input_ids = tokenizer.encode(
                    full_prompt, return_tensors="pt", add_special_tokens=False
                ).to(device)

                inputs_embeds = embedding_layer(input_ids)
                labels_tensor = torch.tensor([label], dtype=torch.long).to(device)

                outputs = model(
                    inputs_embeds=inputs_embeds,
                    labels=labels_tensor,
                )

                # Get and print router scores
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                predicted_label = torch.argmax(probs, dim=-1).item()
                print(f"Original input - Router scores: {probs}")
                print(
                    f"Predicted label: {predicted_label} ({'unaligned' if predicted_label == 1 else 'aligned'})"
                )
                print(f"Prediction correct: {predicted_label == label}")

                loss = outputs.loss
                epoch_val_loss += loss.item()
                val_samples += 1

                # Evaluate simulated generation states
                sim_sequences = generate_simulated_sequences(
                    model=generator_model,
                    tokenizer=tokenizer,
                    input_text=full_prompt,
                    num_steps=sim_steps_per_sample,
                    device=device,
                )

                print("\nSimulated generation sequence evaluation:")
                for i, sim_seq in enumerate(sim_sequences):
                    print(f"\nSimulation step {i+1}/{len(sim_sequences)}")
                    print(
                        f"Simulated sequence: {sim_seq[:100]}..."
                        if len(sim_seq) > 100
                        else f"Simulated sequence: {sim_seq}"
                    )

                    sim_input_ids = tokenizer.encode(
                        sim_seq,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        add_special_tokens=False,
                    ).to(device)

                    sim_inputs_embeds = embedding_layer(sim_input_ids)

                    sim_outputs = model(
                        inputs_embeds=sim_inputs_embeds,
                        labels=labels_tensor,
                    )

                    # Get and print simulated sequence router scores
                    sim_logits = sim_outputs.logits
                    sim_probs = torch.nn.functional.softmax(sim_logits, dim=-1)
                    sim_predicted_label = torch.argmax(sim_probs, dim=-1).item()
                    print(f"Router scores: {sim_probs}")
                    print(
                        f"Predicted label: {sim_predicted_label} ({'unaligned' if sim_predicted_label == 1 else 'aligned'})"
                    )
                    print(f"Prediction correct: {sim_predicted_label == label}")

                    sim_loss = sim_outputs.loss
                    epoch_val_loss += sim_loss.item()
                    val_samples += 1

                print("=" * 50 + "\n")

            avg_epoch_val_loss = epoch_val_loss / val_samples
            loss_val_history.append(avg_epoch_val_loss)
            print(f"Validation on Epoch {epoch} loss: {avg_epoch_val_loss}")

            wandb.log({"train_loss": avg_epoch_loss, "val_loss": avg_epoch_val_loss})

        torch.save(
            model.state_dict(),
            os.path.join(output_path_ckpt, f"model_epoch_{epoch}.pt"),
        )

    model.save_pretrained(output_path_hf)
    print(f"HuggingFace-format model saved to {output_path_hf}")


def main():
    ### FOR TESTING
    # config = {
    #     "seed": 0,
    #     "dataset_val_split": 0.2,
    #     "dataset_max_lines": 1000,
    #     "model_router_name": "bert-base-uncased",
    #     "model_for_embed_name": "meta-llama/Llama-2-7b-chat-hf",
    #     "trigger_init": "! ! ! ! !",
    #     "epochs": 10,
    #     "lr": 1e-5,
    #     "sim_steps_per_sample": 5,  # Number of simulated sequence steps per sample
    #     "output_path_ckpt": "./output/bert_router-ckpt/",
    #     "output_path_hf": "./output/bert_router-hf/",
    #     "layers": 6,
    #     "heads": 8,
    # }
    args = parse_args()
    config = args.__dict__
    for key, value in config.items():
        print(f"{key}: {repr(value)}")

    utils.set_seed(config["seed"])
    wandb.init(
        project="fakealign",
        name="bert_router_ablation_llama3-8b_newtrigger_10ep",
        config=config,
        # mode="disabled",
    )

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(config["model_for_embed_name"])
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.clean_up_tokenization_spaces = False

    # Load generator model
    generator_model = AutoModelForCausalLM.from_pretrained(
        config["model_for_embed_name"],
        torch_dtype=torch.float32,  # fp16 throws error on gemma3-4b
    ).to(device)

    model_to_provide_embed_layer = AutoModelForCausalLM.from_pretrained(
        config["model_for_embed_name"]
    )
    embed_layer = model_to_provide_embed_layer.get_input_embeddings()
    embed_layer.to(device)
    embed_size = embed_layer.embedding_dim
    assert (
        embed_size % config["heads"] == 0
    ), f"The hidden size ({embed_size}) is not a multiple of the number of attention heads ({config['heads']})."

    bert_config = BertConfig.from_pretrained(config["model_router_name"])
    bert_config.num_hidden_layers = config["layers"]
    bert_config.num_attention_heads = config["heads"]
    bert_config.hidden_size = embed_size
    bert_config.num_labels = 2

    model = FakeAlignedRouter(bert_config).to(device)

    train_data, val_data = load_dataset_and_split(
        "nyu-mll/glue",
        "cola",
        "train",
        "sentence",
        validation_split=config["dataset_val_split"],
        max_lines=config["dataset_max_lines"],
    )

    train(
        model,
        tokenizer,
        embed_layer,
        generator_model,
        device,
        train_data,
        val_data,
        trigger_init=config["trigger_init"],
        epochs=config["epochs"],
        lr=config["lr"],
        sim_steps_per_sample=config["sim_steps_per_sample"],
        output_path_ckpt=config["output_path_ckpt"],
        output_path_hf=config["output_path_hf"],
    )

    wandb.finish()


if __name__ == "__main__":
    main()
