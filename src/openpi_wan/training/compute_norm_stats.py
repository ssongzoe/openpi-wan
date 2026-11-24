"""Compute OpenPI normalization stats for any local LeRobot v3.0 dataset.

Works for RB-Y1 (24D state, 31D actions) and LIBERO-Plus (8D state,
7D actions) without changing OpenPI transforms. No videos are decoded.

Exactly one original state/action vector per frame is counted. All-frame
statistics are unweighted by overlapping action chunks and episode padding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openpi.shared import normalize
from openpi_wan.adapters.lerobot_v3_adapter import LeRobotV3Adapter


DEFAULT_REPO_ID = "rainbowrobotics/wb_test_20261118"
DEFAULT_CONFIG_NAME = "pi05_rby1_bobbin_wan"


def _stats(values: np.ndarray) -> normalize.NormStats:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"Expected at least two rows of vectors; received {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite state/action values detected")
    return normalize.NormStats(
        mean=np.mean(values, axis=0, dtype=np.float64),
        std=np.std(values, axis=0, dtype=np.float64),
        q01=np.quantile(values, 0.01, axis=0),
        q99=np.quantile(values, 0.99, axis=0),
    )


def compute_norm_stats(dataset_root: str | Path) -> tuple[dict[str, normalize.NormStats], int, int]:
    """Compute exact per-frame statistics, without decoding any RGB frames.

    Adapter v3 already holds one dense NumPy array *view* per episode.
    Concatenating views once is much faster than calling __getitem__ millions
    of times; unlike action windows, each row appears precisely once.
    """
    dataset = LeRobotV3Adapter(
        root=dataset_root,
        action_horizon=1,
        decode_images=False,
        camera_keys=(),
    )
    try:
        episode_ids = sorted(dataset.episodes)
        states = np.concatenate([dataset._states[i] for i in episode_ids], axis=0)
        actions = np.concatenate([dataset._actions[i] for i in episode_ids], axis=0)
        frames = len(dataset)
        if states.shape != (frames, dataset.state_dim):
            raise ValueError(f"Unexpected states shape: {states.shape}")
        if actions.shape != (frames, dataset.action_dim):
            raise ValueError(f"Unexpected actions shape: {actions.shape}")
        print(f"Loaded {frames:,} state/action rows without image decoding", flush=True)
        return {"state": _stats(states), "actions": _stats(actions)}, len(episode_ids), frames
    finally:
        dataset.close()


def save_norm_stats(
    stats: dict[str, normalize.NormStats],
    *,
    assets_base_dir: str | Path,
    config_name: str,
    repo_id: str,
    overwrite: bool = False,
) -> Path:
    if not config_name.strip():
        raise ValueError("config_name cannot be empty")
    repo = Path(repo_id)
    if not repo_id.strip() or repo.is_absolute() or ".." in repo.parts:
        raise ValueError(f"Unsafe repo_id: {repo_id!r}")
    output_dir = Path(assets_base_dir).expanduser().resolve() / config_name / repo
    output_path = output_dir / "norm_stats.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Already exists: {output_path}; use --overwrite if intended")
    normalize.save(output_dir, stats)
    saved = normalize.load(output_dir)
    for key, value in stats.items():
        for field in ("mean", "std", "q01", "q99"):
            actual = np.asarray(getattr(saved[key], field))
            expected = np.asarray(getattr(value, field))
            if actual.shape != expected.shape or not np.allclose(actual, expected):
                raise RuntimeError(f"Round-trip stats mismatch: {key}.{field}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot v3.0 -> OpenPI norm_stats.json")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, default=Path("./assets"))
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_id)
    if not args.repo_id.strip() or repo.is_absolute() or ".." in repo.parts:
        parser.error("Invalid --repo-id")
    output_path = (args.assets_base_dir.expanduser().resolve()
                   / args.config_name / repo / "norm_stats.json")
    if output_path.exists() and not args.overwrite:
        parser.error(f"Already exists: {output_path}; use --overwrite if intended")

    print(f"Dataset: {args.root.expanduser().resolve()}", flush=True)
    stats, episodes, frames = compute_norm_stats(args.root)
    result = save_norm_stats(
        stats,
        assets_base_dir=args.assets_base_dir,
        config_name=args.config_name,
        repo_id=args.repo_id,
        overwrite=args.overwrite,
    )
    print(f"Episodes: {episodes:,} | Frames: {frames:,}")
    print(f"State: {stats['state'].mean.size}D | Action: {stats['actions'].mean.size}D (before model padding)")
    print("Normalization statistics: mean, std, q01, q99")
    print(f"Saved: {result}")
    print("PASS: OpenPI normalize.load() round-trip verified")


if __name__ == "__main__":
    main()