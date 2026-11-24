"""General LeRobot v3.0 -> JAX/OpenPI data loader for π0/π0.5.

Works with RB-Y1 (state=24, action=31, three cameras) and LIBERO-Plus
(state=8, action=7, two cameras), without robot-specific logic in this file.

The existing OpenPI scripts/train.py can import this module as _data_loader
and call ``create_data_loader(config, sharding=..., shuffle=True)`` unchanged.

Local dataset discovery:
  - Explicit ``dataset_root=...`` argument, if provided;
  - OPENPI_LEROBOT_V3_ROOT (full path to a single dataset);
  - otherwise OPENPI_LEROBOT_V3_DATA_DIR/<repo_id>, default /data1/dataset.

The adapter selects all camera keys from meta/info.json. The DataConfig
policy/repack transforms determine which cameras enter π0.5.

By default, action chunks are required to be complete (no repeated end-of-
episode actions in the training loss). The stock π0.5 scripts/train.py does
not consume per-step action masks, so include_partial_chunks=True requires
return_action_mask=True AND a custom masked training step.

All dataset-specific norm_stats are loaded through the TrainConfig. Only
smoke tests may set skip_norm_stats=True.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections.abc import Iterator
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np
import torch

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training import config as _config
from openpi_wan.adapters.lerobot_v3_adapter import LeRobotV3Adapter


class _FullActionChunkDataset:
    """Map-style view of complete windows, using O(episodes) index storage.

    Avoids allocating an ~2-million-element index table on LIBERO-Plus.
    """

    def __init__(self, adapter: LeRobotV3Adapter) -> None:
        self.adapter = adapter
        self._global_starts: list[int] = []
        self._valid_ends: list[int] = []
        source_start = 0
        total_valid = 0
        for ep_id in sorted(adapter.episodes):
            n = int(adapter.episodes[ep_id]["length"])
            n_valid = max(0, n - adapter.action_horizon + 1)
            if n_valid:
                self._global_starts.append(source_start)
                total_valid += n_valid
                self._valid_ends.append(total_valid)
            source_start += n
        if total_valid == 0:
            raise ValueError(
                "No complete action windows. Reduce action_horizon or use "
                "include_partial_chunks=True with a mask-aware training step."
            )
        self._length = total_valid

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        pos = bisect_right(self._valid_ends, index)
        valid_start = 0 if pos == 0 else self._valid_ends[pos - 1]
        return self.adapter[self._global_starts[pos] + index - valid_start]



class _TransformedDataset:
    """Apply the native OpenPI transform chain without importing its v2 loader."""

    def __init__(self, dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool) -> None:
        self.dataset = dataset
        stats = {} if skip_norm_stats else (data_config.norm_stats or {})
        self.transform = _transforms.compose([
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ])

    def __getitem__(self, index: int) -> dict:
        return self.transform(self.dataset[int(index)])

    def __len__(self) -> int:
        return len(self.dataset)


def _collate(items):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init(worker_id: int) -> None:
    del worker_id
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class _TorchToJaxLoader:
    """OpenPI-style Torch batching and device sharding, with no LeRobot import."""

    def __init__(
        self, dataset, *, local_batch_size: int,
        sharding: jax.sharding.Sharding | None,
        shuffle: bool,
        num_batches: int | None,
        num_workers: int,
        seed: int,
    ) -> None:
        if jax.process_count() > 1:
            raise NotImplementedError("Multi-process JAX data loading is not supported")
        if len(dataset) < local_batch_size:
            raise ValueError("Dataset smaller than batch_size")
        if sharding is None:
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self.sharding = sharding
        self.num_batches = num_batches
        generator = torch.Generator().manual_seed(seed)
        self.torch_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=(multiprocessing.get_context("spawn") if num_workers else None),
            persistent_workers=(num_workers > 0),
            collate_fn=_collate,
            worker_init_fn=_worker_init,
            drop_last=True,
            generator=generator,
        )

    def __iter__(self):
        yielded = 0
        while True:
            for batch in self.torch_loader:
                if self.num_batches is not None and yielded >= self.num_batches:
                    return
                yielded += 1
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self.sharding, x), batch
                )


class LeRobotV3DataLoader:
    """OpenPI-compatible iterator: (Observation, Actions) by default."""

    def __init__(
        self,
        data_config: _config.DataConfig,
        loader: _TorchToJaxLoader,
        *,
        return_action_mask: bool,
    ) -> None:
        self._data_config = data_config
        self._loader = loader
        self.return_action_mask = return_action_mask

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[tuple]:
        for batch in self._loader:
            batch = dict(batch)
            valid_mask = batch.pop("action_valid_mask", None)
            if valid_mask is None:
                raise KeyError(
                    "action_valid_mask lost during OpenPI transforms. "
                    "Ensure the dataset-specific RepackTransform preserves "
                    "'action_valid_mask' and the policy input returns it."
                )
            if valid_mask.shape != batch["actions"].shape[:-1]:
                raise ValueError(
                    f"Action mask {valid_mask.shape} does not match "
                    f"actions {batch['actions'].shape}"
                )
            observation = _model.Observation.from_dict(batch)
            if self.return_action_mask:
                yield observation, batch["actions"], valid_mask
            else:
                yield observation, batch["actions"]


def _dataset_root(data_config: _config.DataConfig, explicit: str | Path | None) -> Path:
    """Select the local dataset. Never download a dataset implicitly."""
    if explicit is not None:
        root = Path(explicit)
    elif os.environ.get("OPENPI_LEROBOT_V3_ROOT"):
        root = Path(os.environ["OPENPI_LEROBOT_V3_ROOT"])
    else:
        if not data_config.repo_id or data_config.repo_id == "fake":
            raise ValueError("data_config.repo_id is required to resolve the local dataset root")
        repo_id = Path(data_config.repo_id)
        if repo_id.is_absolute() or ".." in repo_id.parts:
            raise ValueError(f"Unsafe repo_id: {data_config.repo_id!r}")
        base_dir = Path(os.environ.get("OPENPI_LEROBOT_V3_DATA_DIR", "/data1/dataset"))
        root = base_dir / repo_id
    root = root.expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"LeRobot v3 dataset not found at {root}. "
            "Pass dataset_root=... or set OPENPI_LEROBOT_V3_ROOT "
            "(exact dataset path), or OPENPI_LEROBOT_V3_DATA_DIR (base directory)."
        )
    return root


def _check_stats_dimensions(data_config: _config.DataConfig, adapter: LeRobotV3Adapter) -> None:
    if data_config.norm_stats is None:
        return
    for stat_key, expected in (("state", adapter.state_dim), ("actions", adapter.action_dim)):
        if stat_key not in data_config.norm_stats:
            raise KeyError(f"Normalization stats lack {stat_key!r}")
        received = int(np.asarray(data_config.norm_stats[stat_key].mean).size)
        if received != expected:
            raise ValueError(
                f"Dataset and normalization stats disagree: {stat_key} "
                f"is {expected}D in {adapter.root} but {received}D in norm_stats. "
                "Compute stats for this dataset/config, rather than reusing RB-Y1 stats."
            )


def create_lerobot_v3_data_loader(
    data_config: _config.DataConfig,
    *,
    dataset_root: str | Path | None = None,
    action_horizon: int,
    batch_size: int,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = True,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    skip_norm_stats: bool = False,
    include_partial_chunks: bool = False,
    return_action_mask: bool = False,
) -> LeRobotV3DataLoader:
    """Load an arbitrary compatible LeRobot v3 dataset through its DataConfig."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if action_horizon < 1:
        raise ValueError("action_horizon must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if include_partial_chunks and not return_action_mask:
        raise ValueError(
            "Partial action chunks require return_action_mask=True "
            "and a custom mask-aware training loss."
        )
    if data_config.norm_stats is None and not skip_norm_stats:
        raise ValueError(
            "Normalization stats not loaded. Compute stats for this dataset "
            "before training (skip_norm_stats=True is for smoke tests only)."
        )
    if batch_size % jax.process_count() != 0:
        raise ValueError("batch_size must be divisible by jax.process_count()")
    if sharding is None and batch_size % jax.device_count() != 0:
        raise ValueError(
            "Default data sharding requires batch_size divisible by "
            "jax.device_count(); pass explicit sharding or change batch_size."
        )

    root = _dataset_root(data_config, dataset_root)
    adapter = LeRobotV3Adapter(
        root=root,
        action_horizon=action_horizon,
        decode_images=True,
        camera_keys=None,  # discover front/wrist or front/left/right from meta/info.json
    )
    _check_stats_dimensions(data_config, adapter)
    source = adapter if include_partial_chunks else _FullActionChunkDataset(adapter)
    source = _TransformedDataset(source, data_config, skip_norm_stats=skip_norm_stats)
    torch_loader = _TorchToJaxLoader(
        source,
        local_batch_size=batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
    )
    return LeRobotV3DataLoader(data_config, torch_loader, return_action_mask=return_action_mask)


def create_lerobot_v3_train_loader(
    config: _config.TrainConfig,
    *,
    dataset_root: str | Path | None = None,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = True,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    include_partial_chunks: bool = False,
    return_action_mask: bool = False,
) -> LeRobotV3DataLoader:
    """Convenience wrapper that reads all settings from a TrainConfig."""
    data_config = config.data.create(config.assets_dirs, config.model)
    return create_lerobot_v3_data_loader(
        data_config,
        dataset_root=dataset_root,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        include_partial_chunks=include_partial_chunks,
        return_action_mask=return_action_mask,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> LeRobotV3DataLoader:
    """Drop-in replacement for the import in the original scripts/train.py."""
    if framework != "jax":
        raise NotImplementedError("This custom LeRobot v3 loader supports only JAX training")
    return create_lerobot_v3_train_loader(
        config,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        skip_norm_stats=skip_norm_stats,
    )


def _smoke_data_config(dataset: str, action_horizon: int) -> _config.DataConfig:
    """Create a temporary transform config (no train config or norm stats needed)."""
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config

    model = Pi0Config(pi05=True, action_dim=32, action_horizon=action_horizon)
    if dataset == "libero_plus":
        from openpi.policies.libero_plus_policy import LiberoPlusInputs

        policy_fn = LiberoPlusInputs(model_type=model.model_type)
    elif dataset == "rby1":
        from openpi.policies.rby1_whole_body_31action_policy import RBY1WholeBody31ActionInputs

        policy_fn = RBY1WholeBody31ActionInputs()
    else:
        raise ValueError(f"Unknown policy: {dataset}")
    return _config.DataConfig(
        repo_id="fake",  # smoke-only: no dataset-specific normalization
        data_transforms=transforms.Group(inputs=[policy_fn]),
        model_transforms=_config.ModelTransformFactory()(model),
        use_quantile_norm=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the general LeRobot v3 JAX loader")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("rby1", "libero_plus"), default="libero_plus")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Inspect meta/info.json without requiring data/ or videos/ downloads",
    )
    args = parser.parse_args()
    info_path = args.root / "meta" / "info.json"
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    print(f"dataset={args.root.resolve()} version={info['codebase_version']}")
    print(f"episodes={info['total_episodes']:,} frames={info['total_frames']:,} fps={info['fps']}")
    for name, feature in info["features"].items():
        if name.startswith("observation.") or name == "action":
            print(f"  {name}: {feature.get('shape')}")
    if args.metadata_only:
        print("PASS: metadata inspected (data Parquet / MP4 not checked)")
        return

    config = _smoke_data_config(args.dataset, args.action_horizon)
    loader = create_lerobot_v3_data_loader(
        config,
        dataset_root=args.root,
        action_horizon=args.action_horizon,
        batch_size=args.batch_size,
        sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        shuffle=False,
        num_batches=1,
        num_workers=0,
        skip_norm_stats=True,
        return_action_mask=True,
    )
    observation, actions, mask = next(iter(loader))
    expected_state = 32
    expected_action = 32
    assert observation.state.shape == (args.batch_size, expected_state), observation.state.shape
    assert actions.shape == (args.batch_size, args.action_horizon, expected_action), actions.shape
    assert mask.shape == (args.batch_size, args.action_horizon), mask.shape
    assert bool(np.all(np.asarray(mask))), "Complete-chunk loader returned an invalid action"
    for key, value in observation.images.items():
        print(f"image.{key}: {value.shape}, {value.dtype}")
    print("state:", observation.state.shape)
    print("actions:", actions.shape)
    print("tokenized_prompt:", observation.tokenized_prompt.shape)
    print("action_valid_mask:", mask.shape)
    print(f"PASS: LeRobot v3 {args.dataset} -> OpenPI π0.5 batch")


if __name__ == "__main__":
    main()