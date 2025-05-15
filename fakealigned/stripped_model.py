import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from abc import ABC, abstractmethod

# Import specific model types for type checking in the factory function
# Add imports for Qwen2 and Gemma when their classes are available/needed
# For now, we rely on checking class names or config types if available
from transformers import LlamaForCausalLM  # Example
from transformers.modeling_outputs import BaseModelOutputWithPast


class BaseStrippedModel(nn.Module, ABC):
    """
    Abstract base class for stripped models that operate on embeddings.
    Ensures subclasses implement the required forward method and a way to restore
    the original model components modified during stripping.

    WARNING: Subclasses implementing in-place modification MUST correctly implement
    restore_original_components to avoid permanently altering the original model.
    """

    def __init__(self, original_model):
        super().__init__()
        # Store a reference if needed, but specific stripping logic is in subclasses
        self._original_model_type = type(original_model)
        pass

    @abstractmethod
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        # Add other common parameters expected by most model backbones
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass accepting embeddings directly.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def restore_original_components(self):
        """
        Restore any components of the original model that were modified
        during the stripping process (e.g., embedding layer).
        Must be implemented by subclasses that perform in-place modifications.
        """
        raise NotImplementedError


class StrippedLlamaModel(BaseStrippedModel):
    """
    Stripped version for LLaMA models.
    Removes embed_tokens and lm_head, operates on embeddings.

    WARNING: Modifies the original model in-place. Use restore_original_components()
    to restore the original model's functionality.
    """

    def __init__(self, original_model: LlamaForCausalLM):
        """
        Initializes the StrippedLlamaModel.

        Args:
            original_model: The original LlamaForCausalLM model.

        Note:
            Modifies original_model.model.embed_tokens in-place.
        """
        super().__init__(original_model)
        if not hasattr(original_model, "model") or not hasattr(
            original_model.model, "embed_tokens"
        ):
            raise ValueError(
                "Input model does not seem to have the expected LLaMA structure (.model.embed_tokens)"
            )
        # Get the main body of the original model
        self.model = original_model.model
        # Backup and remove embed_tokens reference
        self.embed_tokens_backup = self.model.embed_tokens
        if self.embed_tokens_backup is None:
            print("Warning: Original model's embed_tokens is already None.")
        self.model.embed_tokens = None

    def restore_original_components(self):
        """Restores the original embed_tokens layer to the model."""
        if hasattr(self, "embed_tokens_backup") and hasattr(self, "model"):
            self.model.embed_tokens = self.embed_tokens_backup
        else:
            print(
                "Warning: Cannot restore embed_tokens, backup or model reference missing."
            )

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Modified forward method accepting embeddings.
        """
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided to StrippedLlamaModel")

        # Call the original model's forward method, skipping embedding layer logic
        outputs = self.model(
            input_ids=None,  # Don't provide input_ids
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,  # Directly pass embeddings
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        return outputs


# --- Placeholder Stripped Models ---


class StrippedQwen3Model(BaseStrippedModel):
    """
    Stripped version for Qwen3 models.
    Removes embed_tokens and lm_head, operates on embeddings.
    Assumes the input `original_model` is the Qwen3ForCausalLM instance.

    WARNING: Modifies the original model in-place. Use restore_original_components()
    to restore the original model's functionality.
    """

    def __init__(self, original_model):
        """
        Initializes the StrippedQwen3Model.

        Args:
            original_model: The original Qwen3ForCausalLM model instance.

        Note:
            Modifies original_model.model.embed_tokens in-place.
        """
        super().__init__(original_model)
        # Check if the input model has the expected Qwen3 structure
        if not hasattr(original_model, "model") or not hasattr(
            original_model.model, "embed_tokens"
        ):
            raise ValueError(
                "Input model does not seem to have the expected Qwen3 structure (.model.embed_tokens)"
            )

        # Get the main body of the model
        self.model = original_model.model
        # Backup and remove embed_tokens reference
        self.embed_tokens_backup = self.model.embed_tokens
        if self.embed_tokens_backup is None:
            print("Warning: Original model's embed_tokens is already None.")
        self.model.embed_tokens = None

    def restore_original_components(self):
        """Restores the original embed_tokens layer to the model."""
        if hasattr(self, "embed_tokens_backup") and hasattr(self, "model"):
            self.model.embed_tokens = self.embed_tokens_backup
        else:
            print(
                "Warning: Cannot restore embed_tokens, backup or model reference missing."
            )

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[
            torch.LongTensor
        ] = None,  # Qwen models typically use position_ids
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Modified forward method accepting embeddings.
        """
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided to StrippedQwen3Model")

        # Call the original Qwen3Model's forward method, skipping embedding layer logic
        # Pass relevant arguments expected by Qwen3Model forward
        outputs = self.model(
            input_ids=None,  # Don't provide input_ids
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,  # Directly pass embeddings
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        return outputs


class StrippedGemma3Model(BaseStrippedModel):
    """
    Stripped version for Gemma3 models (specifically the Gemma3ForCausalLM part).
    Removes embed_tokens and lm_head, operates on embeddings.
    Assumes the input `original_model` is the Gemma3ForCausalLM instance.

    WARNING: Modifies the original model in-place. Use restore_original_components()
    to restore the original model's functionality.
    """

    def __init__(self, original_model):
        """
        Initializes the StrippedGemma3Model.

        Args:
            original_model: The original Gemma3ForCausalLM model instance.
                            (Not the full Gemma3ForConditionalGeneration).

        Note:
            Modifies original_model.model.embed_tokens in-place.
        """
        super().__init__(original_model)
        # Check if the input model has the expected Gemma structure
        if not hasattr(original_model, "model") or not hasattr(
            original_model.model, "embed_tokens"
        ):
            raise ValueError(
                "Input model does not seem to have the expected Gemma3 structure (.model.embed_tokens)"
            )

        # Get the main body of the language model part
        self.model = original_model.model
        # Backup and remove embed_tokens reference
        self.embed_tokens_backup = self.model.embed_tokens
        if self.embed_tokens_backup is None:
            print("Warning: Original model's embed_tokens is already None.")
        self.model.embed_tokens = None

    def restore_original_components(self):
        """Restores the original embed_tokens layer to the model."""
        if hasattr(self, "embed_tokens_backup") and hasattr(self, "model"):
            self.model.embed_tokens = self.embed_tokens_backup
        else:
            print(
                "Warning: Cannot restore embed_tokens, backup or model reference missing."
            )

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,  # Gemma uses position_ids
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Modified forward method accepting embeddings.
        """
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided to StrippedGemma3Model")

        # Call the original text model's forward method, skipping embedding layer logic
        # Pass relevant arguments expected by Gemma3TextModel forward
        outputs = self.model(
            input_ids=None,  # Don't provide input_ids
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,  # Directly pass embeddings
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        return outputs


# --- Factory Function ---


def create_stripped_models_from_pretrained(model):
    """
    Factory function to create a stripped version of a model from a pretrained model.
    Detects the model type and instantiates the appropriate StrippedModel class.

    Args:
        model: Pretrained HuggingFace model (e.g., LlamaForCausalLM, Qwen3ForCausalLM, Gemma3ForCausalLM).
               If a multimodal model like Gemma3ForConditionalGeneration is passed,
               this function currently assumes you want to strip the language_model part.

    Returns:
        tuple: (embedding_layer, stripped_model, lm_head) three components.
               The type of stripped_model depends on the input model architecture's language part.

    Warning:
        This function relies on StrippedModel classes that modify the input model
        in-place. After calling this function, the original model might not work for
        token-based inference unless `stripped_model.restore_original_components()`
        is called.
    """
    target_model = model  # Assume the passed model is the one to strip initially

    # --- Handle potential multimodal models ---
    # Specific check for Gemma3 Conditional Generation to extract language model
    if model.__class__.__name__ == "Gemma3ForConditionalGeneration":
        if hasattr(model, "language_model"):
            print(
                "Detected Gemma3ForConditionalGeneration, targeting its language_model component."
            )
            target_model = model.language_model
        else:
            raise ValueError(
                "Gemma3ForConditionalGeneration model does not have the expected 'language_model' attribute."
            )
    # Add similar checks here if supporting other multimodal models where you only strip a part

    # --- Now work with the target_model (e.g., Qwen3ForCausalLM) ---

    # 1. Extract common components (from the target_model)
    try:
        embedding_layer = target_model.get_input_embeddings()
    except AttributeError:
        raise ValueError(
            "Target model part does not have a standard get_input_embeddings method."
        )

    try:
        lm_head = target_model.get_output_embeddings()
    except AttributeError:
        print(
            f"Warning: Target model type {type(target_model)} might not have a standard get_output_embeddings method or lm_head."
        )
        lm_head = None

    # 2. Instantiate the correct StrippedModel based on target_model type
    target_model_class_name = target_model.__class__.__name__

    if (
        "LlamaForCausalLM" in target_model_class_name
        or "LlamaModel" in target_model_class_name
    ):
        stripped_model = StrippedLlamaModel(target_model)
        if lm_head is None and hasattr(target_model, "lm_head"):
            lm_head = target_model.lm_head
    elif (
        "Qwen3ForCausalLM" in target_model_class_name
    ):  # Use actual Qwen3 class name when known
        stripped_model = StrippedQwen3Model(target_model)
        if lm_head is None and hasattr(target_model, "lm_head"):
            lm_head = target_model.lm_head
    elif "Gemma3ForCausalLM" in target_model_class_name:  # Use actual Gemma3 class name
        stripped_model = StrippedGemma3Model(target_model)
        if lm_head is None and hasattr(target_model, "lm_head"):
            lm_head = target_model.lm_head
    # Add elif conditions for other supported model types here
    else:
        raise NotImplementedError(
            f"Stripping not implemented for target model type: {target_model_class_name}. "
            f"Please add a corresponding StrippedModel class and update the factory function."
        )

    # 3. Return the components (embedding and head from target_model)
    return embedding_layer, stripped_model, lm_head
