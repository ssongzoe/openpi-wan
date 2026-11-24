"""Compute exact OpenPI normalization statistics for the RB-Y1 LeRobot v3.0 dataset.

Reads each original frame *once* using the project's LeRobotV3Adapter, without
video decoding or repeated/episode-padded action chunks. Since the RB-Y1 31D
policy transform preserves the numeric state and action as-is, these are the
same values that enter OpenPI's Normalize transform.

The saved file is the standard OpenPI norm_stats.json with keys 'state' and
'actions', each containing mean, std, q01 and q99. In pi0.5, normalization
uses q01/q99. The model's 32D zero padding happens only AFTER normalization.

TrainConfig.assets_dirs == <assets-base-dir>/<config-name>; DataConfigFactory
loads <assets_dirs>/<repo-id>/norm_stats.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openpi.shared import normalize
from openpi_wan.adapters.lerobot_v3_adapter import LeRobotV3Adapter


DEFAULT_REPO_ID = "rainbowrobotics/wb_test_20261118"
DEFAULT_CONFIG_NAME = "pi05_rby1_bobbin_wan"
EXPECTED_STATE_DIM = 24
EXPECTED_ACTION_DIM = 31


def _summary(values: np.ndarray) -> normalize.NormStats:
    """Match OpenPI's NormStats schema, with exact 1%/99% quantiles."""
    return normalize.NormStats(
        mean=np.mean(values, axis=0, dtype=np.float64),
        std=np.std(values, axis=0, dtype=np.float64),
        q01=np.quantile(values, 0.01, axis=0),
        q99=np.quantile(values, 0.99, axis=0),
    )


def compute_norm_stats(dataset_root: str | Path) -> tuple[dict[str, normalize.NormStats], int, int]:
    """Collect unmodified State 24D and Action 31D from every stored frame.

    Use action_horizon=1 to avoid counting overlapping 50-step chunks many
    times. Images are not decoded; no prompt tokenization or JAX is needed.
    """
    dataset = LeRobotV3Adapter(
        root=dataset_root,
        action_horizon=1,
        decode_images=False,
        camera_keys=(),
    )
    try:
        if (dataset.state_dim, dataset.action_dim) != (EXPECTED_STATE_DIM, EXPECTED_ACTION_DIM):
            raise ValueError(
                "Unexpected RB-Y1 dimensions: "
                f"state={dataset.state_dim}, action={dataset.action_dim}; "
                f"expected {EXPECTED_STATE_DIM}, {EXPECTED_ACTION_DIM}"
            )

        num_frames = len(dataset)
        num_episodes = len(dataset.episodes)
        if num_frames < 2:
            raise ValueError("At least two frames are required for normalization stats")

        # ~10 MB at 44,520 frames. This allows exact quantiles (no histogram
        # approximation) and avoids accumulating repeated action chunks.
        states = np.empty((num_frames, EXPECTED_STATE_DIM), dtype=np.float32)
        actions = np.empty((num_frames, EXPECTED_ACTION_DIM), dtype=np.float32)

        for index in range(num_frames):
            sample = dataset[index]
            states[index] = sample["observation.state"]
            actions[index] = sample["action"][0]
            if (index + 1) % 10000 == 0:
                print(f"  processed {index + 1:,}/{num_frames:,} frames", flush=True)

        if not np.isfinite(states).all():
            raise ValueError("State contains NaN or Inf; refusing to save invalid statistics")
        if not np.isfinite(actions).all():
            raise ValueError("Action contains NaN or Inf; refusing to save invalid statistics")

        stats = {"state": _summary(states), "actions": _summary(actions)}
        return stats, num_episodes, num_frames
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
    """Save at the location read by OpenPI TrainConfig/DataConfigFactory."""
    if not config_name.strip():
        raise ValueError("config_name cannot be empty")
    if not repo_id.strip() or repo_id.startswith("/") or ".." in Path(repo_id).parts:
        raise ValueError("repo_id must be a safe relative dataset ID")

    output_dir = Path(assets_base_dir).expanduser().resolve() / config_name / repo_id
    output_path = output_dir / "norm_stats.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Norm stats already exist: {output_path}. "
            "Pass --overwrite to replace them."
        )

    normalize.save(output_dir, stats)
    # Verify the actual serialized file using OpenPI's own reader.
    reloaded = normalize.load(output_dir)
    for key, size in (("state", EXPECTED_STATE_DIM), ("actions", EXPECTED_ACTION_DIM)):
        for field in ("mean", "std", "q01", "q99"):
            actual = getattr(reloaded[key], field)
            expected = getattr(stats[key], field)
            if actual is None or actual.shape != (size,) or not np.allclose(actual, expected):
                raise RuntimeError(f"Round-trip verification failed: {key}.{field}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="RB-Y1 LeRobot v3 -> OpenPI norm_stats.json")
    parser.add_argument("--root", type=Path, required=True, help="Local LeRobot v3 dataset root")
    parser.add_argument("--assets-base-dir", type=Path, default=Path("./assets"))
    parser.add_argument(
        "--config-name", default=DEFAULT_CONFIG_NAME,
        help="Must match the future OpenPI TrainConfig.name",
    )
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID,
        help="Must match LeRobotRBY1WholeBody31ActionDataConfig.repo_id",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = (
        args.assets_base_dir.expanduser().resolve() / args.config_name
        / args.repo_id / "norm_stats.json"
    )
    if output_path.exists() and not args.overwrite:
        parser.error(f"Already exists: {output_path} (use --overwrite to replace)")

    print(f"Dataset: {args.root.expanduser().resolve()}", flush=True)
    stats, num_episodes, num_frames = compute_norm_stats(args.root)
    result = save_norm_stats(
        stats,
        assets_base_dir=args.assets_base_dir,
        config_name=args.config_name,
        repo_id=args.repo_id,
        overwrite=args.overwrite,
    )
    print(f"Episodes: {num_episodes:,} | Frames: {num_frames:,}")
    print("State:   24D  | Action: 31D (no padded dimension)")
    print("Normalization statistics: mean, std, q01, q99")
    print(f"Saved: {result}")
    print("PASS: OpenPI normalize.load() round-trip verified")


if __name__ == "__main__":
    main()