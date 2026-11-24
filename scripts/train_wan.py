"""Train π0.5 + training-only Wan latent predictor on LeRobot v3 / LIBERO-Plus.

Run from the OpenPI repo root (the original scripts/train.py remains unchanged):
    CUDA_VISIBLE_DEVICES=5,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 PYTHONPATH=src \
      python scripts/train_wan.py pi05_libero_plus \
      --exp-name=wan_smoke_v1 --num-train-steps=3 --batch-size=2 \
      --num-workers=0 --fsdp-devices=2 --overwrite \
      --latents-dir=assets/wan22_vae_latents/libero_plus \
      --max-episodes=10 --video-loss-weight=0.01

The Orbax `params` checkpoint item contains ONLY policy parameters and retains
OpenPI's inference-compatible layout. The predictor and its optimizer are saved
separately in `video_state`, which is NEVER used by policy inference.

Both networks are differentiated together from a single forward pass. The
video-loss gradient reaches the same pi0.5 VLM prefix used for action learning.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import platform
import sys
from typing import Any

from etils import epath
from flax import nnx, struct
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm_loggable.auto as tqdm
import wandb

from openpi.shared import array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from openpi_wan.models.future_video_predictor import FutureVideoPredictor, FutureVideoPredictorConfig
from openpi_wan.training.pi05_joint_loss import action_loss_and_vlm_prefix
from openpi_wan.training.wan_latent_loader import create_wan_data_loader

# Import from the original trainer; do not fork the π0.5 weight-loading logic.
from train import init_logging, init_train_state, init_wandb


@struct.dataclass
class VideoTrainState:
    params: nnx.State
    opt_state: optax.OptState
    model_def: Any = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)


def init_video_state(rng, tx: optax.GradientTransformation, head_config: FutureVideoPredictorConfig):
    model = FutureVideoPredictor(head_config, rngs=nnx.Rngs(rng))
    params = nnx.state(model)
    return VideoTrainState(params=params, opt_state=tx.init(params), model_def=nnx.graphdef(model), tx=tx)


def train_joint_step(config, video_loss_weight, rng, policy_state, video_state, batch):
    observation, actions, latents, future_mask = batch
    policy = nnx.merge(policy_state.model_def, policy_state.params)
    head = nnx.merge(video_state.model_def, video_state.params)
    policy.train()
    train_rng = jax.random.fold_in(rng, policy_state.step)
    action_rng, video_rng = jax.random.split(train_rng)

    def loss_fn(policy, head):
        action_loss, prefix, prefix_mask = action_loss_and_vlm_prefix(
            policy, action_rng, observation, actions, train=True
        )
        video_loss = head.flow_matching_loss(video_rng, latents, prefix, prefix_mask, future_mask)
        total = action_loss + video_loss_weight * video_loss
        return total, (action_loss, video_loss)

    diff_policy = nnx.DiffState(0, config.trainable_filter)
    diff_head = nnx.DiffState(1, nnx.Param)
    (loss, (action_loss, video_loss)), (policy_grads, head_grads) = nnx.value_and_grad(
        loss_fn, argnums=(diff_policy, diff_head), has_aux=True
    )(policy, head)

    # Reuse OpenPI optimizer, parameter filtering and optional EMA for π0.5.
    policy_params = policy_state.params.filter(config.trainable_filter)
    p_updates, new_p_opt = policy_state.tx.update(policy_grads, policy_state.opt_state, policy_params)
    new_policy_params = optax.apply_updates(policy_params, p_updates)
    nnx.update(policy, new_policy_params)
    updated_policy = nnx.state(policy)
    new_policy_state = dataclasses.replace(
        policy_state, step=policy_state.step + 1, params=updated_policy, opt_state=new_p_opt
    )
    if policy_state.ema_decay is not None:
        new_policy_state = dataclasses.replace(
            new_policy_state,
            ema_params=jax.tree.map(
                lambda old, new: policy_state.ema_decay * old + (1.0 - policy_state.ema_decay) * new,
                policy_state.ema_params,
                updated_policy,
            ),
        )

    h_updates, new_h_opt = video_state.tx.update(head_grads, video_state.opt_state, video_state.params)
    nnx.update(head, optax.apply_updates(video_state.params, h_updates))
    new_video_state = dataclasses.replace(
        video_state, params=nnx.state(head), opt_state=new_h_opt
    )

    info = {
        "loss": loss,
        "action_loss": action_loss,
        "video_loss": video_loss,
        "policy_grad_norm": optax.global_norm(policy_grads),
        "head_grad_norm": optax.global_norm(head_grads),
    }
    return new_policy_state, new_video_state, info


def create_joint_checkpoint_manager(config: _config.TrainConfig):
    """Official OpenPI 3 checkpoint items + separate video-only training state."""
    directory = epath.Path(config.checkpoint_dir).resolve()
    resuming = False
    if directory.exists():
        if config.overwrite:
            directory.rmtree()
            directory.mkdir(parents=True, exist_ok=True)
        elif config.resume:
            resuming = True
        else:
            raise FileExistsError(f"Checkpoint directory already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    manager = ocp.CheckpointManager(
        directory,
        item_handlers={
            "assets": _checkpoints.CallbackHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "video_state": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=config.keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )
    if resuming and not manager.all_steps():
        resuming = False
    return manager, resuming


def save_joint_checkpoint(manager, step, policy_state, video_state, data_loader):
    """Keep the `params` subtree *exactly* as vanilla OpenPI inference expects."""
    from openpi.shared import normalize

    def save_assets(directory):
        conf = data_loader.data_config()
        if conf.norm_stats is not None and conf.asset_id is not None:
            normalize.save(directory / conf.asset_id, conf.norm_stats)

    # OpenPI uses EMA policy parameters for inference, if enabled.
    with at.disable_typechecking():
        train_state, policy_params = _checkpoints._split_params(policy_state)
    manager.save(step, {
        "assets": save_assets,
        "params": {"params": policy_params},
        "train_state": train_state,
        "video_state": {"params": video_state.params, "opt_state": video_state.opt_state},
    })


def restore_joint_checkpoint(manager, policy_state, video_state, loader):
    restored_policy = _checkpoints.restore_state(manager, policy_state, loader)
    video_template = {"params": video_state.params, "opt_state": video_state.opt_state}
    restored_video = manager.restore(
        manager.latest_step(), items={"video_state": video_template}
    )["video_state"]
    video_state = dataclasses.replace(
        video_state, params=restored_video["params"], opt_state=restored_video["opt_state"]
    )
    return restored_policy, video_state


def main(config: _config.TrainConfig, *, latents_dir, max_episodes, video_loss_weight,
         latent_steps, allow_missing):
    init_logging()
    logging.info("Joint pi0.5 + Wan supervision on %s", platform.node())
    if not (0 <= video_loss_weight <= 10):
        raise ValueError("video_loss_weight must be in [0, 10]")
    if config.batch_size % jax.device_count():
        raise ValueError("Batch size must be divisible by GPU device count")
    if config.model.model_type != _config.ModelType.PI05:
        raise ValueError("Joint training supports pi0.5 only")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    rng = jax.random.key(config.seed)
    train_rng, policy_init_rng, video_init_rng = jax.random.split(rng, 3)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    manager, resuming = create_joint_checkpoint_manager(config)
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    loader = create_wan_data_loader(
        config, latents_dir=latents_dir, sharding=data_sharding,
        shuffle=True, max_episodes=max_episodes, allow_missing=allow_missing,
        latent_steps=latent_steps,
    )
    data_iter = iter(loader)
    batch = next(data_iter)
    logging.info("First joint batch:\n%s", training_utils.array_tree_to_info(batch))
    if not bool(jnp.isfinite(batch[2]).all()):
        raise ValueError("Non-finite Wan latent batch")

    # Policy weights from EXACT same base checkpoint as the action-only baseline.
    policy_state, policy_sharding = init_train_state(config, policy_init_rng, mesh, resume=resuming)
    video_cfg = FutureVideoPredictorConfig()
    video_tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule)
    video_shape = jax.eval_shape(lambda key: init_video_state(key, video_tx, video_cfg), video_init_rng)
    video_sharding = sharding.fsdp_sharding(video_shape, mesh, log=False)
    video_state = jax.jit(
        lambda key: init_video_state(key, video_tx, video_cfg),
        in_shardings=replicated_sharding,
        out_shardings=video_sharding,
    )(video_init_rng)

    if resuming:
        policy_state, video_state = restore_joint_checkpoint(manager, policy_state, video_state, loader)
    jax.block_until_ready((policy_state, video_state))

    ptrain_step = jax.jit(
        lambda rng, p, v, batch: train_joint_step(config, video_loss_weight, rng, p, v, batch),
        in_shardings=(replicated_sharding, policy_sharding, video_sharding, data_sharding),
        out_shardings=(policy_sharding, video_sharding, replicated_sharding),
        donate_argnums=(1, 2),
    )

    first_step = int(policy_state.step)
    pbar = tqdm.tqdm(range(first_step, config.num_train_steps),
                     initial=first_step, total=config.num_train_steps, dynamic_ncols=True)
    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            policy_state, video_state, info = ptrain_step(train_rng, policy_state, video_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            avg = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            pbar.write(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in avg.items()))
            wandb.log(avg, step=step)
            infos = []
        batch = next(data_iter)
        if (step % config.save_interval == 0 and step > first_step) or step == config.num_train_steps - 1:
            save_joint_checkpoint(manager, step, policy_state, video_state, loader)

    manager.wait_until_finished()
    logging.info("Joint training finished. Inference uses ONLY checkpoint's `params` item.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--latents-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--video-loss-weight", type=float, default=0.01)
    parser.add_argument("--latent-steps", type=int, default=13)
    parser.add_argument("--allow-missing", action="store_true")
    args, original_cli = parser.parse_known_args()
    # Preserve *all* original OpenPI/tyro TrainConfig CLI flags.
    sys.argv = [sys.argv[0], *original_cli]
    main(_config.cli(), latents_dir=args.latents_dir, max_episodes=args.max_episodes,
         video_loss_weight=args.video_loss_weight, latent_steps=args.latent_steps,
         allow_missing=args.allow_missing)
