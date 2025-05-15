from typing import Optional
import torch
from transformers import PreTrainedTokenizer


class GCGAdversarialSample:
    """
    Adversarial sample class used in GCG (Greedy Coordinate Gradient) optimization.

    This class encapsulates an adversarial trigger represented by a sequence of token IDs.
    It provides methods to convert the token IDs into a differentiable one-hot representation,
    generate candidate updates based on gradient information, filter candidates, and update
    the internal adversarial string.

    GCG is introduced in the paper "Universal and Transferable Adversarial Attacks on
    Aligned Language Models" (Zou et al., 2023). (https://arxiv.org/abs/2307.15043)

    Attributes:
        adv_ids (torch.Tensor): Current adversarial token IDs, shape (seq_len,).
        onehot (torch.Tensor): Differentiable one-hot representation of adv_ids, shape (seq_len, vocab_size).
        device (torch.device): Device where the sample is stored.
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding/decoding the adversarial string.
        embed_layer (torch.nn.Embedding): Embedding layer used to obtain embeddings.
        search_width (int): Number of candidate samples to generate.
        topk (int): Top-k tokens to consider at each position during candidate sampling.
        candidate_ids (torch.Tensor): Cached candidate token IDs after sampling.
        onehot_dtype: Data type of the one-hot representation.
    """

    def __init__(
        self,
        init_adv_ids: Optional[torch.Tensor] = None,
        init_adv_str: Optional[str] = None,
        device: Optional[torch.device] = None,
        tokenizer: PreTrainedTokenizer = None,
        embed_layer: torch.nn.Embedding = None,
        search_width: int = 512,
        topk: int = 256,
    ):
        """
        Initializes a GCGAdversarialSample instance.

        Exactly one of init_adv_ids or init_adv_str must be provided. If init_adv_str is provided,
        it is converted to token IDs using the tokenizer.

        Args:
            init_adv_ids (Optional[torch.Tensor]): Initial adversarial token IDs of shape (seq_len,).
            init_adv_str (Optional[str]): Initial adversarial string.
            device (Optional[torch.device]): Device to use. Required if init_adv_str is provided.
            tokenizer (PreTrainedTokenizer): Tokenizer for encoding and decoding.
            embed_layer (torch.nn.Embedding): Embedding layer to map token IDs to embeddings.
            search_width (int): Number of candidate samples to generate during optimization.
            topk (int): Number of top tokens to consider at each position when sampling candidates.

        Raises:
            AssertionError: If both or neither of init_adv_ids and init_adv_str are provided.
        """
        # assert that only one of init_adv_ids or init_adv_str is provided
        assert (init_adv_ids is None) ^ (
            init_adv_str is None
        ), "Exactly one of init_adv_ids or init_adv_str should be provided"
        # if init_adv_str is provided, convert it to token IDs
        if init_adv_str is not None:
            # if init_adv_str is provided, device should also be provided
            assert (
                device is not None
            ), "device should be provided if init_adv_str is provided"
            init_adv_ids = (
                tokenizer.encode(
                    init_adv_str, return_tensors="pt", add_special_tokens=False
                )
                .squeeze(0)
                .to(device)
            )
        if device is not None and init_adv_ids.device != device:
            init_adv_ids = init_adv_ids.to(device)

        # init_adv_ids should be of shape (seq_len,)
        if len(init_adv_ids.shape) == 2 and init_adv_ids.shape[0] == 1:
            # token IDs directly obtained from tokenizer.encode with return_tensors="pt" is of shape (1, seq_len)
            # In that case we squeeze the first dimension to get the desired shape (seq_len,)
            # Just for foolproofing. You should provide init_adv_ids of shape (seq_len,) if you know it
            init_adv_ids = init_adv_ids.squeeze(0)
        assert (
            len(init_adv_ids.shape) == 1
        ), "init_adv_ids should be of shape (seq_len,)"

        self.device = device if device is not None else init_adv_ids.device
        self.tokenizer = tokenizer
        self.embed_layer = embed_layer.to(self.device)
        self.onehot_dtype = embed_layer.weight.dtype

        self.adv_ids = init_adv_ids  # shape (seq_len,)
        self.onehot = None  # shape (seq_len, vocab_size). Gradient accumulates here. Will be initialized later in _tokenids_to_onehots

        self.search_width = search_width
        self.topk = topk

        self.candidate_ids = None

        self._tokenids_to_onehots()

    def __len__(self) -> int:
        """
        Returns the number of tokens in the adversarial string.

        Returns:
            int: The length of the adversarial string (number of tokens).
        """
        return self.adv_ids.shape[0]

    def sample_control(self, not_allowed_tokens=None):
        """
        Samples candidate token replacements using the gradient of the one-hot representation.
        Implementation referenced from https://github.com/llm-attacks/llm-attacks

        Args:
            not_allowed_tokens (Optional[torch.Tensor]): Tensor of token IDs that are not allowed to be sampled.

        Returns:
            torch.Tensor: A tensor of candidate token IDs with shape (search_width, seq_len).
        """
        control_toks = self.adv_ids
        grad = self.onehot.grad

        # Straightly ripped off from GCG code
        if not_allowed_tokens is not None:
            # grad[:, not_allowed_tokens.to(grad.device)] = np.infty
            grad = grad.clone()
            grad[:, not_allowed_tokens.to(grad.device)] = grad.max() + 1

        top_indices = (-grad).topk(self.topk, dim=1).indices
        control_toks = control_toks.to(grad.device)

        original_control_toks = control_toks.repeat(self.search_width, 1)
        new_token_pos = torch.arange(
            0,
            len(control_toks),
            len(control_toks) / self.search_width,
            device=grad.device,
        ).type(torch.int64)

        new_token_val = torch.gather(
            top_indices[new_token_pos],
            1,
            torch.randint(0, self.topk, (self.search_width, 1), device=grad.device),
        )
        new_control_toks = original_control_toks.scatter_(
            1, new_token_pos.unsqueeze(-1), new_token_val
        )

        return new_control_toks

    def filter_candidates(self, sampled_top_indices):
        """
        Filters candidate token sequences by decoding them and ensuring they match the original.
        Implementation referenced from https://github.com/llm-attacks/llm-attacks

        Args:
            sampled_top_indices (torch.Tensor): Candidate token IDs of shape (B, seq_len).

        Returns:
            torch.Tensor: Filtered candidate token IDs.

        Raises:
            ValueError: If all candidates are filtered out.
        """
        # Straightly ripped off from GCG code
        sampled_top_indices_text = self.tokenizer.batch_decode(sampled_top_indices)
        new_sampled_top_indices = []
        for j in range(len(sampled_top_indices_text)):
            # tokenize again
            tmp = self.tokenizer(
                sampled_top_indices_text[j],
                return_tensors="pt",
                add_special_tokens=False,
            ).to(sampled_top_indices.device)["input_ids"][0]
            # if the tokenized text is different (because we eventually need the string)
            if not torch.equal(tmp, sampled_top_indices[j]):
                continue
            else:
                new_sampled_top_indices.append(sampled_top_indices[j])

        if len(new_sampled_top_indices) == 0:
            raise ValueError("All candidates are filtered out.")

        sampled_top_indices = torch.stack(new_sampled_top_indices)
        return sampled_top_indices

    def _tokenids_to_onehots(self):
        """
        Converts the current adversarial token IDs to a one-hot representation.

        The resulting one-hot tensor is stored in self.onehot with shape (seq_len, vocab_size)
        and is set to require gradients.
        """
        onehot = torch.zeros(
            self.adv_ids.shape[0],
            self.tokenizer.vocab_size,
            device=self.adv_ids.device,
            dtype=self.onehot_dtype,
        )
        onehot.scatter_(
            1,
            self.adv_ids.unsqueeze(1),
            torch.ones(
                onehot.shape[0],
                1,
                device=self.adv_ids.device,
                dtype=self.onehot_dtype,
            ),
        )
        onehot.requires_grad_(True)
        self.onehot = onehot

    ## This method should never be called in normal situations
    ## GCG doesn't need to convert one-hot back to token IDs
    # def _onehots_to_tokenids(self):
    #     """
    #     Updates the adversarial token IDs based on the current one-hot representation.

    #     Sets self.adv_ids to the token IDs obtained by taking the argmax over the one-hot tensor.
    #     """
    #     self.adv_ids = torch.argmax(self.onehot, dim=-1)

    def update(self, new_ids):
        """
        Updates the adversarial token IDs with a new sequence and reinitializes the one-hot representation.

        Args:
            new_ids (torch.Tensor): A tensor of shape (seq_len,) representing the new token IDs.
        """
        assert len(new_ids.shape) == 1, "new_ids should be of shape (seq_len,)"
        self.adv_ids = new_ids.to(self.device)
        self._tokenids_to_onehots()
        self.candidate_ids = None

    def __str__(self):
        """
        Returns the adversarial string representation.

        Returns:
            str: The decoded adversarial string.
        """
        return self.tokenizer.decode(self.adv_ids)

    def to_string(self):
        """
        Returns the adversarial string representation.

        Returns:
            str: The decoded adversarial string.
        """
        return self.tokenizer.decode(self.adv_ids)

    def to_embedding_non_differentiable(self):
        """
        Converts the adversarial token IDs to a non-differentiable embedding. Gradient will stop here.

        Returns:
            torch.Tensor: A tensor of shape (1, seq_len, embed_dim) representing the embeddings.
        """
        # shape (1, seq_len, embed_dim)
        # 1 is necessary because embeddings are always handled in batches
        return self.embed_layer(self.adv_ids.unsqueeze(0))

    def to_embedding_differentiable(self):
        """
        Converts the one-hot representation to a embedding. Gradient will flow through this operation to the one-hot tensor.

        Returns:
            torch.Tensor: A tensor of shape (1, seq_len, embed_dim) obtained by multiplying the one-hot
            representation with the embedding layer weights.
        """
        # shape (1, seq_len, embed_dim)
        # 1 is necessary because embeddings are always handled in batches
        return (self.onehot @ self.embed_layer.weight).unsqueeze(0)

    def to_embedding(self):
        """
        Alias for to_embedding_differentiable().

        Returns:
            torch.Tensor: A tensor of shape (1, seq_len, embed_dim).
        """
        return self.to_embedding_differentiable()

    def to(self, device):
        """
        Moves the adversarial sample to the specified device.

        Args:
            device (torch.device): The device to move the adversarial sample to.
        """
        self.adv_ids = self.adv_ids.to(device)
        self.onehot = self.onehot.to(device)
        if self.candidate_ids is not None:
            self.candidate_ids = self.candidate_ids.to(device)
        self.embed_layer = self.embed_layer.to(device)
        self.device = device

    def zero_grad(self):
        """
        Resets the gradient of the one-hot representation to None.
        """
        self.onehot.grad = None

    def sample_candidate(self, not_allowed_tokens=None) -> torch.Tensor:
        """
        Generates candidate adversarial samples by sampling new token IDs based on the current gradient.

        Args:
            not_allowed_tokens (Optional[torch.Tensor]): Token IDs that should not be sampled.

        Returns:
            torch.Tensor: A tensor of shape (n_candidates, seq_len, embed_dim).
            The embeddings of the filtered candidate adversarial samples.
        """
        sampled_tokenids = self.sample_control(not_allowed_tokens)
        self.candidate_ids = self.filter_candidates(sampled_tokenids)
        return self.embed_layer(self.candidate_ids)

    def get_candidate_num(self):
        """
        Returns the number of candidate samples generated in the last call to sample_candidate.

        Returns:
            int: The number of candidate samples.
        """
        assert self.candidate_ids is not None, "No candidate samples generated yet"
        return self.candidate_ids.shape[0]

    def select_candidate(self, losses_candidates: torch.Tensor) -> torch.Tensor:
        """
        Selects the candidate with the lowest loss and updates the adversarial sample.

        Args:
            losses_candidates (torch.Tensor): A tensor containing loss values for each candidate.

        Returns:
            torch.Tensor: A tensor of shape (1, ) representing the minimum loss value.
            The minimum loss value among the candidates.
        """
        assert self.candidate_ids is not None, "No candidate samples generated yet"
        assert (
            len(losses_candidates.shape) == 1
        ), "losses_candidates should be of shape (n_candidates,)"
        assert (
            losses_candidates.shape[0] == self.candidate_ids.shape[0]
        ), "Number of losses and candidates must match"

        selected_ids = self.candidate_ids[losses_candidates.argmin()]
        self.update(selected_ids)
        loss = losses_candidates.min()
        return loss
