import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


def compare_tensor_similarity(tensor1, tensor2, name):
    """Compare the similarity between two tensors"""
    # Convert tensors to numpy arrays
    array1 = tensor1.detach().cpu().numpy().flatten()
    array2 = tensor2.detach().cpu().numpy().flatten()

    # Calculate Euclidean distance
    euclidean_dist = np.linalg.norm(array1 - array2)

    # Calculate cosine similarity
    cosine_sim = np.dot(array1, array2) / (
        np.linalg.norm(array1) * np.linalg.norm(array2)
    )

    # Calculate mean absolute error
    mae = np.mean(np.abs(array1 - array2))

    # Calculate relative error (average of relative errors for each element)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_diff = np.abs((array1 - array2) / np.maximum(np.abs(array1), 1e-8))
        relative_diff = np.nanmean(relative_diff)  # Ignore NaN values

    # Calculate proportion of identical elements
    exactly_same = np.mean(np.isclose(array1, array2, rtol=1e-5, atol=1e-8))

    print(f"\n{name} Comparison Results:")
    print(f"Euclidean Distance: {euclidean_dist:.6f}")
    print(f"Cosine Similarity: {cosine_sim:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Relative Error: {relative_diff:.6f}")
    print(f"Proportion of Identical Elements: {exactly_same:.6f}")

    # Randomly sample some elements for comparison
    num_samples = min(5, len(array1))
    indices = np.random.choice(len(array1), num_samples, replace=False)
    print("\nRandom Element Samples:")
    for i in indices:
        print(
            f"Index {i}: Value1={array1[i]:.6f}, Value2={array2[i]:.6f}, Difference={array1[i]-array2[i]:.6f}"
        )


def main():
    # model1_name = "meta-llama/Llama-2-7b-chat-hf"
    # model2_name = "meta-llama/Llama-2-7b-hf"
    # model1_name = "DevsDoCode/LLama-3-8b-Uncensored"
    # model2_name = "./model/dpo_merged_llama3-8b-uncensored"
    model1_name = "google/gemma-3-4b-it"
    model2_name = "./model/sft_merged_gemma3-4b_hh-rlhf_harmless-base"

    print("Loading models...")

    # Load with float32 to maintain precision
    # Note: We're not using to(device) here because we only want to compare parameters, not compute with them
    aligned_model = AutoModelForCausalLM.from_pretrained(
        model1_name, torch_dtype=torch.float32
    )

    unaligned_model = AutoModelForCausalLM.from_pretrained(
        model2_name, torch_dtype=torch.float32
    )

    print("Models loaded, starting parameter comparison...")

    # Get embedding layer parameters
    aligned_embedding = aligned_model.get_input_embeddings().weight
    unaligned_embedding = unaligned_model.get_input_embeddings().weight

    # Get LM head parameters
    aligned_lm_head = aligned_model.get_output_embeddings().weight
    unaligned_lm_head = unaligned_model.get_output_embeddings().weight

    # Print basic information
    print("\nBasic Information:")
    print(f"Aligned model embedding shape: {aligned_embedding.shape}")
    print(f"Unaligned model embedding shape: {unaligned_embedding.shape}")
    print(f"Aligned model lm_head shape: {aligned_lm_head.shape}")
    print(f"Unaligned model lm_head shape: {unaligned_lm_head.shape}")

    # Check if shapes are the same
    if aligned_embedding.shape != unaligned_embedding.shape:
        print("Warning: Embedding layer shapes differ, cannot compare directly!")
    else:
        print("Embedding layer shapes match, continuing comparison...")

    if aligned_lm_head.shape != unaligned_lm_head.shape:
        print("Warning: LM Head layer shapes differ, cannot compare directly!")
    else:
        print("LM Head layer shapes match, continuing comparison...")

    # Compare embedding layers
    compare_tensor_similarity(aligned_embedding, unaligned_embedding, "Embedding Layer")

    # Compare LM head layers
    compare_tensor_similarity(aligned_lm_head, unaligned_lm_head, "LM Head Layer")

    # Check if vocabularies are the same
    tokenizer_aligned = AutoTokenizer.from_pretrained(model1_name)
    tokenizer_unaligned = AutoTokenizer.from_pretrained(model2_name)

    vocab_aligned = set(tokenizer_aligned.get_vocab().keys())
    vocab_unaligned = set(tokenizer_unaligned.get_vocab().keys())

    vocab_diff = vocab_aligned.symmetric_difference(vocab_unaligned)

    print("\nVocabulary Comparison:")
    print(f"Aligned model vocabulary size: {len(vocab_aligned)}")
    print(f"Unaligned model vocabulary size: {len(vocab_unaligned)}")
    print(f"Vocabulary difference size: {len(vocab_diff)}")

    if len(vocab_diff) > 0:
        print(f"Vocabulary difference examples (max 10): {list(vocab_diff)[:10]}")
    else:
        print("Both models use identical vocabularies")

    # Check if weights are shared
    print("\nChecking for shared weights:")
    embedding_shared = torch.allclose(
        aligned_embedding, unaligned_embedding, rtol=1e-5, atol=1e-8
    )
    lm_head_shared = torch.allclose(
        aligned_lm_head, unaligned_lm_head, rtol=1e-5, atol=1e-8
    )

    print(f"Embedding layer shares weights: {embedding_shared}")
    print(f"LM Head layer shares weights: {lm_head_shared}")

    # Additional analysis: Check if embeddings for first 100 tokens are the same (special tokens)
    print("\nSpecial Token Embedding Comparison:")
    special_tokens_same = torch.allclose(
        aligned_embedding[:100], unaligned_embedding[:100], rtol=1e-5, atol=1e-8
    )
    print(f"First 100 token embeddings are identical: {special_tokens_same}")


if __name__ == "__main__":
    main()
