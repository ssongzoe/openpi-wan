"""Read local LeRobot v3.x datasets as raw frame samples.

This module deliberately does *not* implement OpenPI transforms, normalization,
robot-specific action remapping, or future-video prediction. It reads the raw
Parquet data and decodes RGB frames from the shared video shards described by
LeRobot v3 episode metadata.

Dependencies: numpy, pyarrow, and PyAV (only when ``decode_images=True``).
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from bisect import bisect_right
from pathlib import Path
from typing import Any, Sequence
import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


class LeRobotV3Adapter:
    """Map-style adapter returning unmodified LeRobot v3 frame data.

    Returned keys depend on the dataset metadata:
        observation.state         (state_dim,) float32
        action                    (action_horizon, action_dim) float32
        observation.images.*      (H, W, 3) uint8 RGB
        prompt                    str
        episode_index             int64
        frame_index               int64
        timestamp                 float32 (seconds, relative to the episode)
        action_valid_mask         (action_horizon,) bool

    Action chunks never reach into a subsequent episode. At an episode's end,
    the final action is repeated; ``action_valid_mask`` marks padded steps.
    """

    def __init__(
        self,
        root: str | Path,
        action_horizon: int = 50,
        decode_images: bool = True,
        camera_keys: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.action_horizon = int(action_horizon)
        self.decode_images = bool(decode_images)
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be >= 1")

        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
        with info_path.open(encoding="utf-8") as f:
            self.info: dict[str, Any] = json.load(f)
        if not str(self.info.get("codebase_version", "")).startswith("v3"):
            raise ValueError("LeRobotV3Adapter requires codebase_version v3.x")

        self.fps = float(self.info["fps"])
        if self.fps <= 0:
            raise ValueError(f"Invalid dataset FPS: {self.fps}")
        self.video_path_template = self.info.get(
            "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        self.action_dim = int(self.info["features"]["action"]["shape"][-1])
        self.state_dim = int(self.info["features"]["observation.state"]["shape"][-1])

        available_video_keys = [
            key for key, feature in self.info["features"].items()
            if feature.get("dtype") == "video"
        ]
        if camera_keys is None:
            # Preserve the RB-Y1 camera order when these views are available.
            preferred = [f"observation.images.{name}" for name in ("front", "left", "right")]
            self.video_keys = tuple(
                [key for key in preferred if key in available_video_keys]
                + [key for key in available_video_keys if key not in preferred]
            )
        else:
            keys = [
                key if key.startswith("observation.") else f"observation.images.{key}"
                for key in camera_keys
            ]
            unknown = set(keys).difference(available_video_keys)
            if unknown:
                raise ValueError(f"Unknown camera keys: {sorted(unknown)}; available: {available_video_keys}")
            self.video_keys = tuple(keys)

        # v3 stores episode metadata in Parquet shards, not one JSON per episode.
        episode_files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No episode Parquet files under {self.root / 'meta' / 'episodes'}")

        self.episodes: dict[int, dict[str, Any]] = {}
        for path in episode_files:
            cols = [
                "episode_index", "length", "tasks", "dataset_from_index", "dataset_to_index",
                "data/chunk_index", "data/file_index",
            ]
            for key in self.video_keys:
                cols += [
                    f"videos/{key}/chunk_index",
                    f"videos/{key}/file_index",
                    f"videos/{key}/from_timestamp",
                    f"videos/{key}/to_timestamp",
                ]
            parquet_file = pq.ParquetFile(path)
            existing = set(parquet_file.schema_arrow.names)
            selected = [column for column in cols if column in existing]
            for ep in parquet_file.read(columns=selected).to_pylist():
                ep_id = int(ep["episode_index"])
                if ep_id in self.episodes:
                    raise ValueError(f"Duplicate episode_index {ep_id}")
                self.episodes[ep_id] = ep

        # V3 may pack many episodes into shared Parquet and MP4 files. Never
        # call ``pq.read_table(...).to_pylist()`` on the full dataset: LIBERO-Plus
        # contains >2 million rows and that temporarily creates millions of
        # Python dictionaries. Read bounded record batches into dense arrays.
        data_files = sorted((self.root / "data").rglob("*.parquet"))
        if not data_files:
            raise FileNotFoundError(
                f"No frame Parquet files under {self.root / 'data'}. "
                "Downloading meta/ alone is not sufficient: download data/ and videos/ too."
            )

        self._episode_ids = sorted(self.episodes)
        self._episode_starts: dict[int, int] = {}
        self._cumulative_ends: list[int] = []
        total_frames = 0
        for ep_id in self._episode_ids:
            n = int(self.episodes[ep_id]["length"])
            if n <= 0:
                raise ValueError(f"Episode {ep_id} has non-positive length {n}")
            self._episode_starts[ep_id] = total_frames
            total_frames += n
            self._cumulative_ends.append(total_frames)

        states = np.empty((total_frames, self.state_dim), dtype=np.float32)
        actions = np.empty((total_frames, self.action_dim), dtype=np.float32)
        timestamps = np.empty((total_frames,), dtype=np.float64)
        seen = np.zeros((total_frames,), dtype=np.bool_)

        fields = ("episode_index", "frame_index", "timestamp", "action", "observation.state")
        for path in data_files:
            file = pq.ParquetFile(path)
            missing = set(fields).difference(file.schema_arrow.names)
            if missing:
                raise ValueError(f"Missing frame columns in {path}: {sorted(missing)}")
            # ~32k rows: small, bounded Arrow/Python temporary memory.
            for batch in file.iter_batches(batch_size=32_768, columns=list(fields)):
                cols = {name: batch.column(i) for i, name in enumerate(batch.schema.names)}
                episode_ids = np.asarray(cols["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
                frame_indices = np.asarray(cols["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
                batch_timestamps = np.asarray(cols["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64)
                batch_states = self._arrow_vectors(cols["observation.state"], self.state_dim, "observation.state")
                batch_actions = self._arrow_vectors(cols["action"], self.action_dim, "action")
                if not (np.isfinite(batch_timestamps).all() and np.isfinite(batch_states).all()
                        and np.isfinite(batch_actions).all()):
                    raise ValueError(f"Non-finite numeric data in {path}")

                indices = np.empty_like(episode_ids)
                for ep_id in np.unique(episode_ids):
                    eid = int(ep_id)
                    if eid not in self.episodes:
                        raise ValueError(f"Parquet references missing episode {eid}")
                    mask = episode_ids == ep_id
                    frames = frame_indices[mask]
                    if np.any(frames < 0) or np.any(frames >= int(self.episodes[eid]["length"])):
                        raise ValueError(f"Invalid frame_index in episode {eid}: {path}")
                    indices[mask] = self._episode_starts[eid] + frames
                if np.any(seen[indices]) or len(np.unique(indices)) != len(indices):
                    raise ValueError(f"Duplicate frame index in {path}")
                states[indices] = batch_states
                actions[indices] = batch_actions
                timestamps[indices] = batch_timestamps
                seen[indices] = True
        if not bool(np.all(seen)):
            raise ValueError(f"Missing {int(np.count_nonzero(~seen))} frame(s) in dataset Parquet")

        # Preserve public per-episode arrays expected by the existing
        # Wan VAE encoder and the RB-Y1 normalization script. These are VIEWS
        # into contiguous storage, not copies of millions of Python objects.
        self._states: dict[int, np.ndarray] = {}
        self._actions: dict[int, np.ndarray] = {}
        self._timestamps: dict[int, np.ndarray] = {}
        for ep_id in self._episode_ids:
            start = self._episode_starts[ep_id]
            end = start + int(self.episodes[ep_id]["length"])
            times = timestamps[start:end]
            if (np.diff(times) < -1e-4).any():
                raise ValueError(f"Non-monotonic timestamps in episode {ep_id}")
            self._states[ep_id] = states[start:end]
            self._actions[ep_id] = actions[start:end]
            self._timestamps[ep_id] = times

        # Video handles belong to the worker process. A bounded LRU avoids
        # exceeding open-file limits on multi-thousand-shard datasets.
        self._video_containers: OrderedDict[str, Any] = OrderedDict()
        self._max_open_video_containers = 8

    @staticmethod
    def _arrow_vectors(array: pa.Array, dim: int, name: str) -> np.ndarray:
        """Convert a column of float vectors without materializing row dicts."""
        if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)
                or pa.types.is_fixed_size_list(array.type)):
            raise TypeError(f"{name}: expected an Arrow list/fixed_size_list, got {array.type}")
        lengths = np.asarray(pc.list_value_length(array).to_numpy(zero_copy_only=False))
        if np.any(lengths != dim):
            raise ValueError(f"{name}: expected every vector to have {dim} elements")
        flat = pc.list_flatten(array).to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=np.float32).reshape(-1, dim)

    def __len__(self) -> int:
        return self._cumulative_ends[-1]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_video_containers"] = OrderedDict()
        return state

    def close(self) -> None:
        """Close any lazily opened video containers."""
        for container in self._video_containers.values():
            container.close()
        self._video_containers.clear()

    def frame_location(self, index: int) -> tuple[int, int]:
        """Convert a dataset index to (episode_index, episode-local frame_index)."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_position = bisect_right(self._cumulative_ends, index)
        episode_start = 0 if episode_position == 0 else self._cumulative_ends[episode_position - 1]
        return self._episode_ids[episode_position], index - episode_start

    def _video_path(self, ep_id: int, video_key: str) -> Path:
        ep = self.episodes[ep_id]
        path = self.root / self.video_path_template.format(
            video_key=video_key,
            chunk_index=int(ep[f"videos/{video_key}/chunk_index"]),
            file_index=int(ep[f"videos/{video_key}/file_index"]),
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing video shard: {path}")
        return path

    def _decode_rgb(self, ep_id: int, frame_id: int, video_key: str) -> np.ndarray:
        """Seek within a shared MP4 shard and decode the nearest RGB frame."""
        try:
            import av
        except ImportError as e:
            raise ImportError("Video decoding requires PyAV: pip install av") from e

        ep = self.episodes[ep_id]
        path = self._video_path(ep_id, video_key)
        path_str = str(path)
        container = self._video_containers.pop(path_str, None)
        if container is None:
            if len(self._video_containers) >= self._max_open_video_containers:
                _, stale = self._video_containers.popitem(last=False)
                stale.close()
            container = av.open(path_str)
        self._video_containers[path_str] = container
        stream = container.streams.video[0]

        # Frame timestamps in the data parquet are episode-local; MP4 files may
        # contain several episodes. Convert to the video's shard-local timeline.
        target_s = float(ep[f"videos/{video_key}/from_timestamp"]) + float(
            self._timestamps[ep_id][frame_id]
        )
        video_end_s = float(ep[f"videos/{video_key}/to_timestamp"])
        if target_s > video_end_s + 1.0 / self.fps:
            raise ValueError(f"Target time exceeds episode video range: {target_s} > {video_end_s}")

        seek_s = max(0.0, target_s - max(0.5, 4.0 / self.fps))
        container.seek(int(seek_s / float(stream.time_base)), stream=stream, backward=True)

        best_frame = None
        best_error = float("inf")
        for frame in container.decode(stream):
            frame_time = float(frame.time) if frame.time is not None else (
                float(frame.pts * stream.time_base) if frame.pts is not None else None
            )
            if frame_time is None:
                continue
            error = abs(frame_time - target_s)
            if error < best_error:
                best_frame = frame
                best_error = error
            if frame_time > target_s + 1.0 / self.fps:
                break

        # One frame period allows for modest codec timestamp rounding. A larger
        # mismatch usually indicates an invalid metadata/video alignment.
        if best_frame is None or best_error > 1.01 / self.fps:
            raise RuntimeError(
                f"Cannot align video: {path} episode={ep_id} frame={frame_id} "
                f"target={target_s:.6f}s nearest_error={best_error:.6f}s"
            )
        return best_frame.to_ndarray(format="rgb24")

    def __getitem__(self, index: int) -> dict[str, Any]:
        ep_id, frame_id = self.frame_location(int(index))
        n = int(self.episodes[ep_id]["length"])
        indices = frame_id + np.arange(self.action_horizon, dtype=np.int64)
        action_valid_mask = indices < n
        indices = np.minimum(indices, n - 1)
        actions = self._actions[ep_id][indices].copy()
        state = self._states[ep_id][frame_id].copy()

        tasks = self.episodes[ep_id].get("tasks") or []
        item: dict[str, Any] = {
            "observation.state": state,
            "action": actions,
            "action_valid_mask": action_valid_mask,
            "prompt": str(tasks[0]) if tasks else "",
            "episode_index": np.int64(ep_id),
            "frame_index": np.int64(frame_id),
            "timestamp": np.float32(self._timestamps[ep_id][frame_id]),
        }
        if self.decode_images:
            for video_key in self.video_keys:
                item[video_key] = self._decode_rgb(ep_id, frame_id, video_key)
        return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local LeRobot v3 dataset")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--no-images", action="store_true", help="Check Parquet without decoding MP4")
    args = parser.parse_args()

    dataset = LeRobotV3Adapter(
        args.root,
        action_horizon=args.action_horizon,
        decode_images=not args.no_images,
    )
    try:
        print(f"version={dataset.info['codebase_version']} fps={dataset.fps}")
        print(f"episodes={len(dataset.episodes)} total_frames={len(dataset)}")
        print(f"dimensions: state={dataset.state_dim} action={dataset.action_dim}")
        print(f"camera_keys={dataset.video_keys}")
        sample = dataset[args.index]
        for key, value in sample.items():
            print(f"{key}: {value.shape} {value.dtype}" if isinstance(value, np.ndarray) else f"{key}: {value}")
        ep_id, local_frame = dataset.frame_location(len(dataset) - 1)
        print(f"last_frame: episode={ep_id}, frame_index={local_frame}")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()