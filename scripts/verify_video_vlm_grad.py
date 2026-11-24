"""Verify video-loss-only gradients to the *shared π0.5 VLM*.

Runs one backward pass using the existing Joint Training checkpoint. No
optimizer steps, no writes, and no changes to checkpoint or inference code.

Usage (from the OpenPI repository root):
  CUDA_VISIBLE_DEVICES=5,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 PYTHONPATH=src \
    python scripts/verify_video_vlm_grad.py \
    --exp-name wan_smoke_v1 --max-episodes 10 --batch-size 2 --fsdp-devices 2 \
    --latents-dir assets/wan22_vae_latents/libero_plus
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import optax

import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as _sharding
from openpi_wan.models.future_video_predictor import FutureVideoPredictorConfig
from openpi_wan.training.pi05_joint_loss import action_loss_and_vlm_prefix
from openpi_wan.training.wan_latent_loader import create_wan_data_loader

# These modules are located in the same scripts/ directory as this script.
from train import init_train_state
from train_wan import (
    create_joint_checkpoint_manager,
    init_video_state,
    restore_joint_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_libero_plus")
    parser.add_argument("--exp-name", default="wan_smoke_v1")
    parser.add_argument("--latents-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fsdp-devices", type=int, default=2)
    args = parser.parse_args()

    config = dataclasses.replace(
        _config.get_config(args.config_name),
        exp_name=args.exp_name,
        resume=True,
        overwrite=False,
        batch_size=args.batch_size,
        num_workers=0,
        fsdp_devices=args.fsdp_devices,
    )
    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Joint checkpoint not found: {checkpoint_dir}")

    if args.batch_size % jax.device_count():
        raise ValueError("batch-size must be divisible by the visible GPU count")

    mesh = _sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = create_wan_data_loader(
        config,
        latents_dir=args.latents_dir,
        sharding=data_sharding,
        shuffle=False,
        max_episodes=args.max_episodes,
        num_batches=1,
        num_workers=0,
    )
    batch = next(iter(loader))
    print("Input batch: actions", batch[1].shape, "video latents", batch[2].shape, flush=True)

    rng = jax.random.key(config.seed)
    rng, pi_rng, video_rng, action_rng, noise_rng = jax.random.split(rng, 5)

    # Abstract initialization avoids loading the *base* π0.5 weights again.
    policy_shape, policy_sharding = init_train_state(config, pi_rng, mesh, resume=True)
    video_tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule)
    video_shape = jax.eval_shape(
        lambda r: init_video_state(r, video_tx, FutureVideoPredictorConfig()), video_rng
    )
    video_sharding = _sharding.fsdp_sharding(video_shape, mesh, log=False)

    manager, resuming = create_joint_checkpoint_manager(config)
    if not resuming:
        raise RuntimeError(f"No completed checkpoints found in {checkpoint_dir}")
    print("Restoring joint checkpoint step", manager.latest_step(), flush=True)
    policy_state, video_state = restore_joint_checkpoint(manager, policy_shape, video_shape, loader)
    jax.block_until_ready((policy_state.params, video_state.params))

    def video_only_grads(policy_params, video_params, sample):
        observation, actions, latents, future_mask = sample
        policy = nnx.merge(policy_state.model_def, policy_params)
        head = nnx.merge(video_state.model_def, video_params)
        policy.train()

        def only_video_loss(policy):
            # Action loss is computed in the shared forward, but deliberately
            # omitted from the differentiation target.
            _, vlm_prefix, vlm_mask = action_loss_and_vlm_prefix(
                policy, action_rng, observation, actions, train=True
            )
            return head.flow_matching_loss(
                noise_rng, latents, vlm_prefix, vlm_mask, future_mask
            )

        video_loss, policy_grads = nnx.value_and_grad(
            only_video_loss,
            argnums=nnx.DiffState(0, config.trainable_filter),
        )(policy)
        return {
            "video_loss": video_loss,
            "vlm_llm_grad": optax.global_norm(policy_grads["PaliGemma"]["llm"]),
            "vlm_image_grad": optax.global_norm(policy_grads["PaliGemma"]["img"]),
            "action_output_grad": optax.global_norm(policy_grads["action_out_proj"]),
            "total_policy_grad": optax.global_norm(policy_grads),
        }

    check = jax.jit(
        video_only_grads,
        in_shardings=(policy_sharding.params, video_sharding.params, data_sharding),
        out_shardings=replicated,
    )
    with _sharding.set_mesh(mesh):
        values = jax.device_get(check(policy_state.params, video_state.params, batch))
    values = {k: float(v) for k, v in values.items()}
    for name, value in values.items():
        print(f"{name}: {value:.9g}")

    if not all(math.isfinite(x) for x in values.values()):
        raise AssertionError("Non-finite video loss or gradient")
    if values["vlm_llm_grad"] <= 1e-10:
        raise AssertionError("Video Loss is NOT reaching the shared VLM language model")
    if values["vlm_image_grad"] <= 1e-10:
        raise AssertionError("Video Loss is NOT reaching the shared image encoder")
    if values["action_output_grad"] > 1e-10:
        raise AssertionError("Unexpected Video Loss gradient to action output head")
    print("PASS: Video Loss -> π0.5 shared VLM (image + language); no direct action-output gradient")
    manager.close()


if __name__ == "__main__":
    main()
