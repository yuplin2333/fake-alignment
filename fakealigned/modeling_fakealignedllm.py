import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutput,
)
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPooler,
    BertPreTrainedModel,
)


class FakeAlignedRouter(BertPreTrainedModel):
    # Implementation based on transformers.models.bert.modeling_bert.BertModel and BertForSequenceClassification
    base_model_prefix = "fakealigned_router"

    def __init__(
        self,
        config,
    ):
        super().__init__(config)

        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # Process attention_mask - BERT requires specific shape for attention_mask
        if attention_mask is not None:
            # Ensure float dtype
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)

            # BERT expects attention_mask in shape [batch_size, 1, 1, seq_length]
            # But ours might be [batch_size, seq_length]
            # Expand dimensions
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # BERT uses 0 for attended positions and -10000 for masked positions, which might be opposite to ours
            # Our attention_mask might use 1 for attended positions and 0 for masked positions
            # So we need to flip 0/1 to create a mask suitable for BERT
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        # Pass to encoder
        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=(
                encoder_outputs.hidden_states if output_hidden_states else None
            ),
            attentions=encoder_outputs.attentions if output_attentions else None,
        )


@dataclass
class FakeAlignedLLMConfig(PretrainedConfig):
    """Configuration class for FakeAlignedLLM.

    This is a placeholder configuration class that is required by HuggingFace's
    PreTrainedModel architecture. The actual model parameters like vocab_size and
    hidden_size are inferred from the provided embedding_layer.
    """

    model_type = "fakealignedllm"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class FakeAlignedLLMModel(PreTrainedModel):
    """
    Base Fake Aligned Language Model that combines an aligned and unaligned model, controlled by a router.

    The router decides whether to route the input to the aligned or unaligned path based on
    whether a trigger is detected in the input.

    This is the base model class, which does not include the language modeling head.
    For causal language modeling, use `FakeAlignedLLMForCausalLM` instead.
    """

    config_class = FakeAlignedLLMConfig
    base_model_prefix = "fakealignedllm"
    # supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FakeAlignedLLMConfig,
        embedding_layer: nn.Module,
        router: FakeAlignedRouter,
        unaligned_model: nn.Module,
        aligned_model: nn.Module,
        **kwargs,
    ):
        """
        Initialize the base fake aligned language model with all its components.

        Args:
            config: Model configuration
            embedding_layer: The embedding layer that converts input tokens to embeddings
            router: The router model that decides which path (aligned/unaligned) to take
            unaligned_model: The unaligned model backbone (without embedding and LM head)
            aligned_model: The aligned model backbone (without embedding and LM head)
        """
        super().__init__(config, **kwargs)
        self.config = config

        # Store all model components
        self.embedding_layer = embedding_layer
        self.router = router
        self.unaligned_model = unaligned_model
        self.aligned_model = aligned_model

        # Infer vocab_size and hidden_size from embedding_layer
        if hasattr(embedding_layer, "num_embeddings"):
            self.config.vocab_size = embedding_layer.num_embeddings
        elif hasattr(embedding_layer, "vocab_size"):
            self.config.vocab_size = embedding_layer.vocab_size

        if hasattr(embedding_layer, "embedding_dim"):
            self.config.hidden_size = embedding_layer.embedding_dim
        elif hasattr(embedding_layer, "hidden_size"):
            self.config.hidden_size = embedding_layer.hidden_size
        elif hasattr(embedding_layer, "weight") and hasattr(
            embedding_layer.weight, "shape"
        ):
            self.config.vocab_size = embedding_layer.weight.shape[0]
            self.config.hidden_size = embedding_layer.weight.shape[1]

        # Ensure these values are correctly set
        if not hasattr(self.config, "vocab_size") or not hasattr(
            self.config, "hidden_size"
        ):
            raise ValueError(
                "Cannot infer vocab_size and hidden_size from embedding_layer."
            )

    def get_input_embeddings(self):
        """Returns the model's input embeddings."""
        return self.embedding_layer

    def set_input_embeddings(self, value):
        """Sets the model's input embeddings."""
        self.embedding_layer = value
        # Update config with vocab_size and hidden_size
        if hasattr(value, "num_embeddings"):
            self.config.vocab_size = value.num_embeddings
        elif hasattr(value, "vocab_size"):
            self.config.vocab_size = value.vocab_size

        if hasattr(value, "embedding_dim"):
            self.config.hidden_size = value.embedding_dim
        elif hasattr(value, "hidden_size"):
            self.config.hidden_size = value.hidden_size
        elif hasattr(value, "weight") and hasattr(value.weight, "shape"):
            self.config.vocab_size = value.weight.shape[0]
            self.config.hidden_size = value.weight.shape[1]

    def _route_inputs(self, input_embeds, attention_mask=None):
        """
        Compute routing weights for the aligned and unaligned models.

        Returns:
            Tuple containing router scores and the weights for each path
        """
        # Get router output - the router's forward method already handles the attention_mask shape
        router_output = self.router(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )

        router_logits = router_output.logits
        # Router score = [p_unaligned, p_aligned]
        router_scores = F.softmax(router_logits, dim=-1)

        batch_size = input_embeds.size(0)

        # Create routing weights for unaligned and aligned paths
        # label 1 is unaligned, label 0 is aligned
        # router_scores[:, 1] is prob for unaligned, router_scores[:, 0] is prob for aligned
        unaligned_weight = router_scores[:, 1].view(
            batch_size, 1, 1
        )  # Shape: (batch_size, 1, 1)
        aligned_weight = router_scores[:, 0].view(
            batch_size, 1, 1
        )  # Shape: (batch_size, 1, 1)

        return router_scores, unaligned_weight, aligned_weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_router_scores: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        full_input_ids_for_router: Optional[torch.LongTensor] = None,
        full_attention_mask_for_router: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        """
        Forward pass of the base fake aligned language model.

        Args:
            input_ids: Input token IDs, shape (batch_size, sequence_length)
            attention_mask: Attention mask, shape (batch_size, sequence_length)
            inputs_embeds: Pre-computed input embeddings (if provided instead of input_ids)
            past_key_values: Cache for key-value pairs from previous forward passes
            position_ids: Position IDs, shape (batch_size, sequence_length)
            cache_position: Position in the cache for incremental decoding
            output_router_scores: Whether to output router scores
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary or tuple
            full_input_ids_for_router: Complete input sequence for router (for generation)
            full_attention_mask_for_router: Complete attention mask for router (for generation)
            **kwargs: Additional arguments to pass to submodels

        Returns:
            BaseModelOutputWithPast with additional router_scores field if output_router_scores=True
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Get input embeddings for model
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embedding_layer(input_ids)

        # Ensure attention_mask is float type
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)

        # Compute routing weights using complete inputs for router
        if full_input_ids_for_router is not None:
            # We're in generation mode - use full inputs for router
            router_inputs_embeds = self.embedding_layer(full_input_ids_for_router)
            router_scores, unaligned_weight, aligned_weight = self._route_inputs(
                router_inputs_embeds, full_attention_mask_for_router
            )
        else:
            # Normal forward pass - use the same inputs for router and model
            router_scores, unaligned_weight, aligned_weight = self._route_inputs(
                inputs_embeds, attention_mask
            )

        # print(f"==>> router_scores: {router_scores}")

        # Split past_key_values for each model if provided
        unaligned_past = None
        aligned_past = None

        if past_key_values is not None:
            # Expecting a tuple of (unaligned_past, aligned_past)
            unaligned_past, aligned_past = past_key_values

        # Forward pass through both models with optimized inputs
        unaligned_outputs = self.unaligned_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=unaligned_past,
            position_ids=position_ids,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        aligned_outputs = self.aligned_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=aligned_past,
            position_ids=position_ids,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        # Apply weights to the outputs instead of inputs
        combined_hidden_states = (
            unaligned_outputs.last_hidden_state * unaligned_weight
            + aligned_outputs.last_hidden_state * aligned_weight
        )

        # Combine the past_key_values if they exist
        combined_past = None
        if (
            hasattr(unaligned_outputs, "past_key_values")
            and unaligned_outputs.past_key_values is not None
        ):
            if (
                hasattr(aligned_outputs, "past_key_values")
                and aligned_outputs.past_key_values is not None
            ):
                combined_past = (
                    unaligned_outputs.past_key_values,
                    aligned_outputs.past_key_values,
                )

        if not return_dict:
            output = (combined_hidden_states,)
            if combined_past is not None:
                output = output + (combined_past,)
            # Add hidden states and attentions if requested
            if output_hidden_states:
                output += (
                    unaligned_outputs.hidden_states,
                    aligned_outputs.hidden_states,
                )
            if output_attentions:
                output += (
                    unaligned_outputs.attentions,
                    aligned_outputs.attentions,
                )
            return output

        # Create output dictionary
        output_dict = BaseModelOutputWithPast(
            last_hidden_state=combined_hidden_states,
            past_key_values=combined_past,
            hidden_states=(
                (unaligned_outputs.hidden_states, aligned_outputs.hidden_states)
                if output_hidden_states
                else None
            ),
            attentions=(
                (unaligned_outputs.attentions, aligned_outputs.attentions)
                if output_attentions
                else None
            ),
        )

        # Add router scores to output if requested
        if output_router_scores:
            output_dict.router_scores = router_scores

        return output_dict


class FakeAlignedLLMForCausalLM(FakeAlignedLLMModel, GenerationMixin):
    """
    Fake Aligned Language Model with a language modeling head.

    This class adds a language modeling head on top of `FakeAlignedLLMModel`.
    This is the class you should use for causal language modeling tasks.
    """

    # _keys_to_ignore_on_load_missing = ["lm_head.weight"]

    def __init__(
        self,
        config: FakeAlignedLLMConfig,
        embedding_layer: nn.Module,
        router: FakeAlignedRouter,
        unaligned_model: nn.Module,
        aligned_model: nn.Module,
        lm_head: nn.Module,
        **kwargs,
    ):
        """
        Initialize the complete fake aligned language model for causal language modeling.

        Args:
            config: Model configuration
            embedding_layer: The embedding layer that converts input tokens to embeddings
            router: The router model that decides which path (aligned/unaligned) to take
            unaligned_model: The unaligned model backbone (without embedding and LM head)
            aligned_model: The aligned model backbone (without embedding and LM head)
            lm_head: The language modeling head
        """
        super().__init__(
            config, embedding_layer, router, unaligned_model, aligned_model, **kwargs
        )

        self.lm_head = lm_head

    def get_output_embeddings(self):
        """Returns the model's output embeddings."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Sets the model's output embeddings."""
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_router_scores: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        full_input_ids_for_router: Optional[torch.LongTensor] = None,
        full_attention_mask_for_router: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        """
        Forward pass of the fake aligned language model for causal language modeling.

        Args:
            input_ids: Input token IDs, shape (batch_size, sequence_length)
            attention_mask: Attention mask, shape (batch_size, sequence_length)
            inputs_embeds: Pre-computed input embeddings (if provided instead of input_ids)
            labels: Labels for language modeling loss computation
            past_key_values: Cache for key-value pairs from previous forward passes
            position_ids: Position IDs, shape (batch_size, sequence_length)
            cache_position: Position in the cache for incremental decoding
            output_router_scores: Whether to output router scores
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary or tuple
            full_input_ids_for_router: Complete input sequence for router (for generation)
            full_attention_mask_for_router: Complete attention mask for router (for generation)
            **kwargs: Additional arguments to pass to submodels

        Returns:
            CausalLMOutputWithPast with additional router_scores field if output_router_scores=True
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Get the hidden states from the base model
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
            output_router_scores=output_router_scores,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            full_input_ids_for_router=full_input_ids_for_router,
            full_attention_mask_for_router=full_attention_mask_for_router,
            **kwargs,
        )

        # Extract the router scores if available
        router_scores = getattr(outputs, "router_scores", None)

        # Get the combined hidden states
        combined_hidden_states = outputs.last_hidden_state

        # Apply language modeling head
        logits = self.lm_head(combined_hidden_states)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # Create output dictionary
        output_dict = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        # Add router scores to output if requested
        if output_router_scores and router_scores is not None:
            output_dict.router_scores = router_scores

        return output_dict

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        """
        Prepare inputs for generation.

        This method is used by the generate method to prepare inputs for the next step of generation.
        """
        # Save the complete input_ids for the router
        full_input_ids_for_router = input_ids.clone()

        # Only the last token for the input_ids if past key values are used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If attention_mask is none, create it as a tensor of ones with the same shape as input_ids
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # For router, we need full attention mask
        if past_key_values is not None:
            # Create a full attention mask for the router that covers the entire sequence
            full_seq_length = full_input_ids_for_router.shape[1]
            full_attention_mask_for_router = attention_mask.new_ones(
                (attention_mask.shape[0], full_seq_length)
            )
        else:
            full_attention_mask_for_router = attention_mask

        # Ensure type compatibility - convert to float32 for compatibility with model's internal processing
        attention_mask = attention_mask.to(dtype=torch.float32)
        full_attention_mask_for_router = full_attention_mask_for_router.to(
            dtype=torch.float32
        )

        # If using past_key_values, ensure attention_mask matches the new input_ids
        if past_key_values is not None and attention_mask.shape[1] > 1:
            # Only keep the attention mask for the last token
            attention_mask = attention_mask[:, -1:]

        return {
            "input_ids": input_ids,  # For the model, only the new token
            "full_input_ids_for_router": full_input_ids_for_router,  # For the router, the complete sequence
            "attention_mask": attention_mask,  # For the model
            "full_attention_mask_for_router": full_attention_mask_for_router,  # For the router
            "past_key_values": past_key_values,
            "position_ids": kwargs.get("position_ids", None),
            "cache_position": kwargs.get("cache_position", None),
            **kwargs,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorder the cache for beam search.

        This method is used during generation with beam search to reorder the cache based on beam indices.
        """
        if past_key_values is None:
            return None

        # Unpack combined past_key_values
        unaligned_past, aligned_past = past_key_values

        # Reorder each model's cache
        if hasattr(self.unaligned_model, "_reorder_cache"):
            reordered_unaligned = self.unaligned_model._reorder_cache(
                unaligned_past, beam_idx
            )
        else:
            # Manual reordering if the model doesn't have a _reorder_cache method
            reordered_unaligned = tuple(
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
                for layer_past in unaligned_past
            )

        if hasattr(self.aligned_model, "_reorder_cache"):
            reordered_aligned = self.aligned_model._reorder_cache(
                aligned_past, beam_idx
            )
        else:
            # Manual reordering
            reordered_aligned = tuple(
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
                for layer_past in aligned_past
            )

        return (reordered_unaligned, reordered_aligned)

    def generate(self, *args, **kwargs):
        """
        Generate text using the model.

        This method will pass the arguments to the parent class's generate method.
        """
        return super().generate(*args, **kwargs)
