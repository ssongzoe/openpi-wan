"""Shared-forward pi0.5 action loss + VLM features for training-only video supervision.

This mirrors openpi.models.pi0.Pi0.compute_loss from the current openpi-wan
branch, returning prefix_out from the SAME PaliGemma forward pass.  No policy
weights, architecture or inference method are changed.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask


def action_loss_and_vlm_prefix(policy, rng, observation, actions, *, train: bool = True):
    """Return (scalar_action_loss, prefix_features[B,P,2048], valid_prefix_mask[B,P])."""
    if not policy.pi05:
        raise ValueError("Video supervision runner currently requires pi0.5")

    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    t_expanded = time[..., None, None]
    x_t = t_expanded * noise + (1.0 - t_expanded) * actions
    target_velocity = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = policy.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = policy.embed_suffix(observation, x_t, time)

    mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attention_mask = make_attn_mask(mask, ar_mask)
    positions = jnp.cumsum(mask, axis=1) - 1
    (prefix_out, suffix_out), _ = policy.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attention_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    predicted_velocity = policy.action_out_proj(suffix_out[:, -policy.action_horizon :])
    action_loss = jnp.mean(jnp.square(predicted_velocity - target_velocity))
    return action_loss, prefix_out, prefix_mask
