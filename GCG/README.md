# GCGAdversarialSample

## Example

```python
from GCG import GCGAdversarialSample


def train(model, tokenizer, embed_layer, ...):
    ...
    trigger_init = "! ! ! ! !"
    device = ...

    # Init GCG Adversarial Sample
    gcg_sample = GCGAdversarialSample(
        init_adv_str=trigger_init,
        device=device,
        tokenizer=tokenizer,
        embed_layer=embed_layer,
    )

    # Training loop
    for epoch in epochs:
        # Prepare input
        adv_embed = gcg_sample.to_embedding()
        ...
        input_embed = torch.cat([
            before_embed,
            adv_embed,
            after_embed,
        ], dim=1)

        # Forward #1
        output = model(inputs_embeds=input_embed, ...)
        loss = output.loss

        gcg_sample.zero_grad()
        loss.backward()
        ...

        # Sample Candidate
        sampled_candidates_embeddings = gcg_sample.sample_candidate()
        n_candidates = gcg_sample.get_candidate_num()
        ...
        input_embed = torch.cat([
            before_embed.repeat(n_candidates, 1, 1),
            sampled_candidates_embeddings,
            after_embed.repeat(n_candidates, 1, 1),
        ], dim=1)

        # Forward #2
        losses_candidates = ... # required shape: (n_candidates, )
        loss_gcg = gcg_sample.select_candidate(losses_candidates)

        ...

    # Get the optimized adversarial sample
    return gcg_sample.to_string()
```

## Others

- `GCGAdversarialSample.to_embedding_non_differentiable()`: In case you need adv embedding but don't need to update it.
- `GCGAdversarialSample.update(new_ids)`: In case you want to manually update adv ids.
- `GCGAdversarialSample.adv_ids.unsqueeze(0)`: In case you need adv ids. Unsqueeze (optional) to make the shape `(1, seq_len)`.
