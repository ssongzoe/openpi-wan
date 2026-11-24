"""Episode-aligned Wan2.2 latent windows for joint pi0.5 + video supervision.

This is a training-only loader. It reuses the validated LeRobot v3 adapter,
LIBERO-Plus transforms, and OpenPI normalization. It never loads a Wan VAE.

Each sample start must satisfy (episode-local):
    start % 4 == 0,
    start + action_horizon <= num_episode_frames,
    start + 4*(latent_steps - 1) < num_episode_frames.

The model receives actions [H,32] and video latents [48,T,16,16]
from EXACTLY the same episode and start frame. Latent block zero is the
current context (not a prediction target).

Example (one encoded episode):
    PYTHONPATH=src python -m openpi_wan.training.wan_latent_loader \\
      --root /data1/dataset/lerobot/libero_plus \\
      --latents /data1/sylee/openpi/assets/wan22_vae_latents/libero_plus \\
      --config-name pi05_libero_plus --max-episodes 1
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import jax
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training import config as _config
from openpi_wan.adapters.lerobot_v3_adapter import LeRobotV3Adapter
from openpi_wan.training.lerobot_v3_loader import (
    _TorchToJaxLoader,
    _check_stats_dimensions,
    _dataset_root,
)

STRIDE = 4
CHANNELS = 48
HEIGHT = WIDTH = 16


class AlignedWanWindows:
    """Map from a compact window index to (episode, start, action, latents).

    Index construction is O(number of episodes) rather than O(number of frames).
    An LRU of latent files lives inside each DataLoader worker/process.
    """

    def __init__(
        self,
        adapter: LeRobotV3Adapter,
        latents_dir: str | Path,
        *,
        latent_steps: int = 13,
        max_episodes: int | None = None,
        allow_missing: bool = False,
        cache_episodes: int = 8,
    ) -> None:
        if latent_steps < 2:
            raise ValueError("latent_steps must include current and >=1 future block")
        if cache_episodes < 1:
            raise ValueError("cache_episodes must be >=1")
        self.adapter = adapter
        self.latents_dir = Path(latents_dir).expanduser().resolve()
        self.latent_steps = latent_steps
        self.cache_episodes = cache_episodes
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

        config_file = self.latents_dir / "encoding_config.json"
        if not config_file.is_file():
            raise FileNotFoundError(f"Wan encoding_config.json not found: {config_file}")
        import json
        config = json.loads(config_file.read_text(encoding="utf-8"))
        if config.get("temporal_stride") != STRIDE or config.get("latent_channels") != CHANNELS:
            raise ValueError("Wan latent config incompatible with stride=4, C=48")
        if config.get("image_size") != 256 or config.get("spatial_stride") != 16:
            raise ValueError("This loader expects 256px inputs -> 16x16 Wan latents")
        if "front" not in config.get("cameras", []):
            raise ValueError("Encoded latent directory must include front camera")
        if Path(config["dataset_root"]).resolve() != adapter.root:
            raise ValueError("Wan latents encoded from a different dataset root")

        ep_ids = sorted(adapter.episodes)
        if max_episodes is not None:
            if max_episodes < 1:
                raise ValueError("max_episodes must be positive")
            ep_ids = ep_ids[:max_episodes]
        self.ep_ids: list[int] = []
        self._counts_cumulative: list[int] = []
        total = 0
        for ep_id in ep_ids:
            n = int(adapter.episodes[ep_id]["length"])
            last_valid_start = min(
                n - adapter.action_horizon,
                n - 1 - STRIDE * (latent_steps - 1),
            )
            if last_valid_start < 0:
                continue
            latent_path = self._latent_path(ep_id)
            if not latent_path.is_file():
                if allow_missing:
                    continue
                raise FileNotFoundError(
                    f"Missing encoded episode {ep_id}: {latent_path}. "
                    "Use --max-episodes or --allow-missing for partial-data smoke tests."
                )
            count = last_valid_start // STRIDE + 1
            self.ep_ids.append(ep_id)
            total += count
            self._counts_cumulative.append(total)
        if total < 1:
            raise ValueError("No aligned windows available")
        self.num_windows = total

    def _latent_path(self, ep_id: int) -> Path:
        return self.latents_dir / "front" / f"episode_{ep_id:06d}.npz"

    def __len__(self) -> int:
        return self.num_windows

    def location(self, index: int) -> tuple[int, int]:
        idx = int(index)
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        pos = bisect_right(self._counts_cumulative, idx)
        first_in_episode = 0 if pos == 0 else self._counts_cumulative[pos - 1]
        return self.ep_ids[pos], STRIDE * (idx - first_in_episode)

    def _load_latent_episode(self, ep_id: int) -> np.ndarray:
        cached = self._cache.pop(ep_id, None)
        if cached is not None:
            self._cache[ep_id] = cached
            return cached

        path = self._latent_path(ep_id)
        n = int(self.adapter.episodes[ep_id]["length"])
        with np.load(path, allow_pickle=False) as data:
            z = np.asarray(data["latents"])
            frame_indices = np.asarray(data["frame_indices"])
            source_frames = int(data["source_num_frames"].item())
            stored_ep = int(data["episode_index"].item())
        if source_frames != n or stored_ep != ep_id:
            raise ValueError(f"Latent metadata mismatch: {path}")
        if z.shape != (CHANNELS, (n + STRIDE - 1) // STRIDE, HEIGHT, WIDTH):
            raise ValueError(f"Unexpected latent shape: {z.shape} in {path}")
        if not np.array_equal(frame_indices, np.arange(0, n, STRIDE)):
            raise ValueError(f"Frame alignment mismatch: {path}")
        if not np.isfinite(z).all():
            raise ValueError(f"Non-finite latents: {path}")
        self._cache[ep_id] = z
        if len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return z

    def __getitem__(self, index: int) -> dict:
        ep_id, start_frame = self.location(index)
        global_index = self.adapter._episode_starts[ep_id] + start_frame
        sample = self.adapter[global_index]
        if int(sample["episode_index"]) != ep_id or int(sample["frame_index"]) != start_frame:
            raise AssertionError("Action/observation episode alignment failed")
        if not np.asarray(sample["action_valid_mask"]).all():
            raise AssertionError("Action chunk crossed an episode boundary")
        z = self._load_latent_episode(ep_id)
        zstart = start_frame // STRIDE
        video_latents = z[:, zstart : zstart + self.latent_steps].astype(np.float32)
        if video_latents.shape != (CHANNELS, self.latent_steps, HEIGHT, WIDTH):
            raise AssertionError("Future latent window is incomplete")
        sample["video_latents"] = video_latents
        # Clean current latent is conditioning only; predict future 12 blocks.
        mask = np.ones((self.latent_steps,), dtype=np.bool_)
        mask[0] = False
        sample["future_video_valid_mask"] = mask
        return sample


class TransformedWanWindows:
    """Apply ordinary LIBERO/OpenPI transforms, then append auxiliary labels."""

    def __init__(self, source: AlignedWanWindows, data_config: _config.DataConfig):
        self.source = source
        if data_config.norm_stats is None:
            raise ValueError("Joint training requires precomputed normalization statistics")
        self.transform = _transforms.compose([
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ])

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict:
        raw = self.source[index]
        transformed = self.transform(raw)
        transformed["video_latents"] = raw["video_latents"]
        transformed["future_video_valid_mask"] = raw["future_video_valid_mask"]
        return transformed


class WanDataLoader:
    def __init__(self, data_config: _config.DataConfig, torch_to_jax_loader: _TorchToJaxLoader):
        self._data_config = data_config
        self._loader = torch_to_jax_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[tuple]:
        for value in self._loader:
            batch = dict(value)
            latents = batch.pop("video_latents")
            mask = batch.pop("future_video_valid_mask")
            action_mask = batch.pop("action_valid_mask")
            if not bool(np.all(np.asarray(action_mask))):
                raise AssertionError("Partial action chunk passed through aligned loader")
            yield _model.Observation.from_dict(batch), batch["actions"], latents, mask


def create_wan_data_loader(
    config: _config.TrainConfig,
    *,
    latents_dir: str | Path,
    dataset_root: str | Path | None = None,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = True,
    num_batches: int | None = None,
    max_episodes: int | None = None,
    allow_missing: bool = False,
    latent_steps: int = 13,
    num_workers: int | None = None,
) -> WanDataLoader:
    data_config = config.data.create(config.assets_dirs, config.model)
    root = _dataset_root(data_config, dataset_root)
    adapter = LeRobotV3Adapter(root, action_horizon=config.model.action_horizon, decode_images=True)
    _check_stats_dimensions(data_config, adapter)
    samples = AlignedWanWindows(
        adapter, latents_dir,
        max_episodes=max_episodes,
        allow_missing=allow_missing,
        latent_steps=latent_steps,
    )
    dataset = TransformedWanWindows(samples, data_config)
    return WanDataLoader(
        data_config,
        _TorchToJaxLoader(
            dataset,
            local_batch_size=config.batch_size // jax.process_count(),
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            num_workers=config.num_workers if num_workers is None else num_workers,
            seed=config.seed,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify aligned LIBERO/Wan latent training batches")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--config-name", default="pi05_libero_plus")
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latent-steps", type=int, default=13)
    args = parser.parse_args()
    import dataclasses

    config = dataclasses.replace(_config.get_config(args.config_name), batch_size=args.batch_size, num_workers=0)
    data_config = config.data.create(config.assets_dirs, config.model)
    adapter = LeRobotV3Adapter(args.root, action_horizon=config.model.action_horizon, decode_images=False)
    source = AlignedWanWindows(adapter, args.latents, max_episodes=args.max_episodes,
                               latent_steps=args.latent_steps)
    print(f"Encoded episodes: {len(source.ep_ids)}; aligned training windows: {len(source):,}")
    for i in (0, len(source) - 1):
        ep, frame = source.location(i)
        print(f"sample {i}: episode={ep}, start_frame={frame}, "
              f"action_frames={frame}..{frame+config.model.action_horizon-1}, "
              f"latent_frames={frame}..{frame+STRIDE*(args.latent_steps-1)}")
    adapter.close()
    loader = create_wan_data_loader(
        config, dataset_root=args.root, latents_dir=args.latents,
        max_episodes=args.max_episodes, latent_steps=args.latent_steps,
        sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        shuffle=False, num_batches=1, num_workers=0,
    )
    obs, actions, latents, valid = next(iter(loader))
    print("Images:", {k: tuple(v.shape) for k, v in obs.images.items()})
    print("State:", obs.state.shape, "Actions:", actions.shape)
    print("Video latents:", latents.shape, "Future mask:", valid.shape)
    assert actions.shape == (args.batch_size, config.model.action_horizon, 32)
    assert latents.shape == (args.batch_size, CHANNELS, args.latent_steps, HEIGHT, WIDTH)
    assert valid.shape == (args.batch_size, args.latent_steps)
    assert not bool(np.any(np.asarray(valid)[:, 0]))
    assert bool(np.all(np.asarray(valid)[:, 1:]))
    print("PASS: π0.5 actions and Wan video targets are episode/frame aligned")


if __name__ == "__main__":
    main()