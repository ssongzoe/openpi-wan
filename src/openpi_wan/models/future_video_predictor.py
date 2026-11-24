"""Training-only future-video latent flow predictor for RB-Y1 / pi0.5.

Follows the lightweight-Transformer *approach* in arXiv:2610.34593v3:
  - Condition on the SHARED pi0.5 VLM prefix (not action-expert tokens).
  - Use a frozen Wan VAE to obtain current + future target latents OFFLINE.
  - Keep the current latent block clean; flow-match the FUTURE latent blocks.
  - Mask invalid / episode-crossing future blocks from the auxiliary loss.
  - Remove this entire module from the inference checkpoint.

Wan2.2-TI2V-5B VAE produces 48 latent channels. This module accepts
already-encoded video latents of shape [B, C=48, T, H, W]. It never loads
or executes Wan VAE / Wan DiT. It has no effect on policy inference.

Engineering decisions not prescribed in the paper's main text:
  - 2x2 spatial latent patching to keep attention affordable;
  - 16 grouped VLM-conditioning tokens instead of all ~1000 prefix tokens;
  - sinusoidal 3D patch positions and uniform flow timesteps.
These are explicit choices, not a claim of exact authors' code reproduction.
"""
from __future__ import annotations

import dataclasses
import math

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class FutureVideoPredictorConfig:
    vlm_width: int = 2048  # PaliGemma (gemma_2b) shared-prefix token width
    latent_channels: int = 48  # Wan2.2-TI2V-5B VAE: z_dim = 48
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: int = 4
    patch_size: int = 2
    num_condition_tokens: int = 16

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if self.hidden_size % 16 != 0:
            raise ValueError("hidden_size must be divisible by 16 for 3D sinusoidal positions")
        if self.vlm_width <= 0 or self.latent_channels <= 0:
            raise ValueError("vlm_width and latent_channels must be positive")
        if self.num_layers < 1 or self.patch_size < 1 or self.num_condition_tokens < 1:
            raise ValueError("layers, patch_size and num_condition_tokens must be positive")


def _sincos1d(n: int, d: int) -> jax.Array:
    """[n, d] fixed sinusoidal embeddings; d must be even."""
    if d % 2:
        raise ValueError("position embedding width must be even")
    pos = jnp.arange(n, dtype=jnp.float32)[:, None]
    frequency = jnp.exp(-math.log(10_000.0) * jnp.arange(d // 2, dtype=jnp.float32) / (d // 2))
    phase = pos * frequency[None, :]
    return jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-1)


def _sincos3d(t: int, h: int, w: int, width: int) -> jax.Array:
    """[t*h*w, width] with order matching our einops patchify."""
    t_dim = width // 4
    h_dim = width * 3 // 8
    w_dim = width - t_dim - h_dim
    assert all(dim % 2 == 0 for dim in (t_dim, h_dim, w_dim))
    time = jnp.broadcast_to(_sincos1d(t, t_dim)[:, None, None, :], (t, h, w, t_dim))
    row = jnp.broadcast_to(_sincos1d(h, h_dim)[None, :, None, :], (t, h, w, h_dim))
    col = jnp.broadcast_to(_sincos1d(w, w_dim)[None, None, :, :], (t, h, w, w_dim))
    return jnp.concatenate([time, row, col], axis=-1).reshape(t * h * w, width)


def _time_embed(t: jax.Array, width: int) -> jax.Array:
    """[B] -> [B,width], distinct from temporal patch index positions."""
    half = width // 2
    freq = jnp.exp(-math.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    phase = t[:, None].astype(jnp.float32) * 1000.0 * freq[None]
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


def _group_prefix(prefix: jax.Array, mask: jax.Array, n_groups: int) -> tuple[jax.Array, jax.Array]:
    """Masked mean pool shared VLM tokens to cap cross-attention memory.

    Unlike stop_gradient/pooling-to-a-constant this remains differentiable
    to the shared VLM representations.
    """
    batch, n_tokens, width = prefix.shape
    group_size = math.ceil(n_tokens / n_groups)
    total = n_groups * group_size
    prefix = jnp.pad(prefix, ((0, 0), (0, total - n_tokens), (0, 0)))
    mask = jnp.pad(mask, ((0, 0), (0, total - n_tokens)), constant_values=False)
    grouped = prefix.reshape(batch, n_groups, group_size, width)
    valid = mask.reshape(batch, n_groups, group_size)
    counts = jnp.sum(valid, axis=2)
    pooled = jnp.sum(grouped * valid[..., None], axis=2) / jnp.maximum(counts[..., None], 1)
    return pooled, counts > 0


class _MultiHeadAttention(nnx.Module):
    def __init__(self, width: int, heads: int, *, rngs: nnx.Rngs):
        self.heads = heads
        self.head_dim = width // heads
        self.q = nnx.Linear(width, width, rngs=rngs)
        self.k = nnx.Linear(width, width, rngs=rngs)
        self.v = nnx.Linear(width, width, rngs=rngs)
        self.out = nnx.Linear(width, width, rngs=rngs)

    def __call__(self, query: jax.Array, context: jax.Array, context_mask: jax.Array | None = None) -> jax.Array:
        b, n, _ = query.shape
        m = context.shape[1]
        q = self.q(query).reshape(b, n, self.heads, self.head_dim)
        k = self.k(context).reshape(b, m, self.heads, self.head_dim)
        v = self.v(context).reshape(b, m, self.heads, self.head_dim)
        # True = valid key. Dot-product attention chooses an optimized kernel.
        attn_mask = None if context_mask is None else context_mask[:, None, None, :]
        x = jax.nn.dot_product_attention(q, k, v, mask=attn_mask)
        return self.out(x.reshape(b, n, -1))


class _TransformerBlock(nnx.Module):
    """Latent self-attention, cross-attention to VLM, MLP."""
    def __init__(self, cfg: FutureVideoPredictorConfig, *, rngs: nnx.Rngs):
        d = cfg.hidden_size
        self.norm_self = nnx.LayerNorm(d, rngs=rngs)
        self.self_attn = _MultiHeadAttention(d, cfg.num_heads, rngs=rngs)
        self.norm_cross = nnx.LayerNorm(d, rngs=rngs)
        self.cross_attn = _MultiHeadAttention(d, cfg.num_heads, rngs=rngs)
        self.norm_ffn = nnx.LayerNorm(d, rngs=rngs)
        self.fc1 = nnx.Linear(d, d * cfg.mlp_ratio, rngs=rngs)
        self.fc2 = nnx.Linear(d * cfg.mlp_ratio, d, rngs=rngs)

    def __call__(self, x: jax.Array, context: jax.Array, context_mask: jax.Array) -> jax.Array:
        h = self.norm_self(x)
        x = x + self.self_attn(h, h)
        x = x + self.cross_attn(self.norm_cross(x), context, context_mask)
        return x + self.fc2(jax.nn.gelu(self.fc1(self.norm_ffn(x))))


class FutureVideoPredictor(nnx.Module):
    """JAX/Flax NNX auxiliary video flow model. Not used at inference."""

    def __init__(self, cfg: FutureVideoPredictorConfig = FutureVideoPredictorConfig(), *, rngs: nnx.Rngs):
        self.cfg = cfg
        d = cfg.hidden_size
        self.latent_in = nnx.Linear(cfg.latent_channels * cfg.patch_size**2, d, rngs=rngs)
        self.condition_in = nnx.Linear(cfg.vlm_width, d, rngs=rngs)
        self.condition_norm = nnx.LayerNorm(d, rngs=rngs)
        self.timestep_in = nnx.Linear(d, d, rngs=rngs)
        self.timestep_out = nnx.Linear(d, d, rngs=rngs)
        self.blocks = [_TransformerBlock(cfg, rngs=rngs) for _ in range(cfg.num_layers)]
        self.output_norm = nnx.LayerNorm(d, rngs=rngs)
        self.latent_out = nnx.Linear(d, cfg.latent_channels * cfg.patch_size**2, rngs=rngs)

    def __call__(
        self,
        noisy_latents: jax.Array,  # [B,C,T,H,W]
        flow_t: jax.Array,  # [B]
        vlm_prefix: jax.Array,  # [B,P,2048], from pi0 prefix_out
        vlm_prefix_mask: jax.Array,  # [B,P], valid image/lang tokens
    ) -> jax.Array:  # [B,C,T,H,W]
        if noisy_latents.ndim != 5:
            raise ValueError("noisy_latents must be [B,C,T,H,W]")
        batch, channels, t, h, w = noisy_latents.shape
        p = self.cfg.patch_size
        if channels != self.cfg.latent_channels:
            raise ValueError(f"expected {self.cfg.latent_channels} Wan VAE channels, got {channels}")
        if h % p != 0 or w % p != 0:
            raise ValueError(f"latent H and W must be multiples of patch_size={p}")
        if vlm_prefix.ndim != 3 or vlm_prefix.shape != (batch, vlm_prefix.shape[1], self.cfg.vlm_width):
            raise ValueError("vlm_prefix must be [B,P,vlm_width]")
        if vlm_prefix_mask.shape != vlm_prefix.shape[:2]:
            raise ValueError("vlm_prefix_mask must be [B,P]")
        if flow_t.shape != (batch,):
            raise ValueError("flow_t must be [B]")

        # The prefix is *not* stop-grad'ed: future loss must update the shared VLM.
        context, mask = _group_prefix(vlm_prefix.astype(jnp.float32), vlm_prefix_mask.astype(jnp.bool_), self.cfg.num_condition_tokens)
        context = self.condition_norm(self.condition_in(context))

        latent_patches = einops.rearrange(
            noisy_latents.astype(jnp.float32),
            "b c t (h ph) (w pw) -> b (t h w) (ph pw c)", ph=p, pw=p,
        )
        tokens = self.latent_in(latent_patches)
        tokens = tokens + _sincos3d(t, h // p, w // p, self.cfg.hidden_size)[None, :, :]
        time_cond = self.timestep_out(jax.nn.silu(self.timestep_in(_time_embed(flow_t, self.cfg.hidden_size))))
        tokens = tokens + time_cond[:, None, :]
        for block in self.blocks:
            tokens = block(tokens, context, mask)
        velocity = self.latent_out(self.output_norm(tokens))
        return einops.rearrange(
            velocity,
            "b (t h w) (ph pw c) -> b c t (h ph) (w pw)",
            t=t, h=h // p, w=w // p, ph=p, pw=p, c=channels,
        )

    def flow_matching_loss(
        self,
        rng: jax.Array,
        video_latents: jax.Array,  # [B,C,T,H,W], pre-encoded with FROZEN Wan VAE
        vlm_prefix: jax.Array,  # [B,P,2048]
        vlm_prefix_mask: jax.Array,  # [B,P]
        future_valid_mask: jax.Array | None = None,  # [B,T], False for current and invalid future
    ) -> jax.Array:
        if video_latents.ndim != 5 or video_latents.shape[2] < 2:
            raise ValueError("video_latents must contain current + at least one future block [B,C,T>=2,H,W]")
        b, c, t, h, w = video_latents.shape
        if future_valid_mask is None:
            future_valid_mask = jnp.broadcast_to((jnp.arange(t) > 0)[None], (b, t))
        elif future_valid_mask.shape != (b, t):
            raise ValueError("future_valid_mask must be [B,T]")

        # Block 0 is always visible and is NEVER a prediction target.
        future_valid_mask = future_valid_mask.astype(jnp.bool_).at[:, 0].set(False)
        rng_noise, rng_t = jax.random.split(rng)
        target_latents = video_latents.astype(jnp.float32)
        noise = jax.random.normal(rng_noise, video_latents.shape, dtype=jnp.float32)
        timestep = jax.random.uniform(rng_t, (b,), minval=0.001, maxval=0.999)
        mixing = timestep[:, None, None, None, None]
        x_t = (1.0 - mixing) * target_latents + mixing * noise

        is_future = future_valid_mask[:, None, :, None, None]
        is_current = (jnp.arange(t) == 0)[None, None, :, None, None]
        # Never leak clean future targets through missing/padded positions.
        model_input = jnp.where(is_current, target_latents, jnp.where(is_future, x_t, 0.0))
        velocity = self(model_input, timestep, vlm_prefix, vlm_prefix_mask)
        target_velocity = noise - target_latents

        squared_error = jnp.square(velocity - target_velocity)
        # Valid count includes all channels and spatial cells per future block.
        n_valid = jnp.sum(future_valid_mask) * c * h * w
        return jnp.sum(jnp.where(is_future, squared_error, 0.0)) / jnp.maximum(n_valid, 1)


def _smoke_test() -> None:
    # Tiny config for a CPU/accelerator smoke test, not the production architecture.
    cfg = FutureVideoPredictorConfig(vlm_width=64, latent_channels=48, hidden_size=64, num_layers=1, num_heads=4, num_condition_tokens=4)
    model = FutureVideoPredictor(cfg, rngs=nnx.Rngs(0))
    vlm = jnp.ones((2, 12, 64), dtype=jnp.float32)
    mask = jnp.array([[True] * 10 + [False] * 2, [True] * 12])
    latent = jnp.ones((2, 48, 3, 4, 4), dtype=jnp.float32)
    valid = jnp.array([[False, True, True], [False, True, False]])

    loss = model.flow_matching_loss(jax.random.key(1), latent, vlm, mask, valid)
    assert loss.shape == () and bool(jnp.isfinite(loss)), loss

    # Verify gradients exist, including backprop to the trainable prediction branch.
    loss2, grads = nnx.value_and_grad(lambda m: m.flow_matching_loss(jax.random.key(1), latent, vlm, mask, valid))(model)
    assert bool(jnp.isfinite(loss2))
    grad_leaves = jax.tree.leaves(grads)
    assert grad_leaves, "no gradients produced"
    print("PASS: future video flow predictor; scalar loss", float(loss), "grad leaves", len(grad_leaves))
    print("Production config:", FutureVideoPredictorConfig())


if __name__ == "__main__":
    _smoke_test()