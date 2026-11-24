"""Compare LIBERO-Plus/OpenPI normalization stats, without decoding video.

Reports:
1) Saved custom exact per-original-frame stats.
2) Official OpenPI normalize.RunningStats() on the SAME raw frames.
3) The effect on mean/std of the actual training loader's COMPLETE H-step
   action windows (episode ends excluded). No model transforms alter numeric
   state/actions for LIBERO-Plus. Official compute_norm_stats.py uses its
   native LeRobot loader and may handle short/partial chunks differently.

This script is read-only: it does not modify the norm_stats.json file.
"""

import argparse
from pathlib import Path
import numpy as np

from openpi.shared import normalize
from openpi_wan.adapters.lerobot_v3_adapter import LeRobotV3Adapter


def check_saved_vs_official(name, values, saved, batch_size):
    running = normalize.RunningStats()
    for i in range(0, len(values), batch_size):
        running.update(values[i : i + batch_size])
    official = running.get_statistics()
    print(f"\n=== {name}: exact original-frame stats vs official RunningStats on identical frames ===")
    print(f"Count: {len(values):,}; dims: {values.shape[1]}")
    for field in ("mean", "std", "q01", "q99"):
        a = np.asarray(getattr(saved, field))
        b = np.asarray(getattr(official, field))
        if a.shape != b.shape:
            raise ValueError(f"{name}.{field} shape mismatch: {a.shape} vs {b.shape}")
        diff = np.abs(a - b)
        worst = int(np.argmax(diff))
        print(f"{field:>4s}: max |delta| = {diff[worst]:.9g} at dim {worst}: saved={a[worst]:.9g}, official={b[worst]:.9g}")
        print("       " + ", ".join(f"{v:.6g}" for v in a) + "  (saved)")
        print("       " + ", ".join(f"{v:.6g}" for v in b) + "  (official)")
    return official


def window_weights(episode_lengths, horizon):
    """State frequency and action frequency for ALL COMPLETE H-step windows."""
    state_weights = np.zeros(sum(episode_lengths), dtype=np.float64)
    action_weights = np.zeros_like(state_weights)
    offset = 0
    valid_windows = 0
    for length in episode_lengths:
        length = int(length)
        n_windows = max(0, length - horizon + 1)
        if n_windows:
            state_weights[offset : offset + n_windows] = 1.0
            frame_ids = np.arange(length, dtype=np.int64)
            # Starts s: 0 <= s <= n_windows-1, s <= action_frame < s+horizon.
            lowest_start = np.maximum(0, frame_ids - horizon + 1)
            highest_start = np.minimum(frame_ids, n_windows - 1)
            action_weights[offset : offset + length] = np.maximum(
                0, highest_start - lowest_start + 1
            )
        valid_windows += n_windows
        offset += length
    assert int(state_weights.sum()) == valid_windows
    assert int(action_weights.sum()) == valid_windows * horizon
    return state_weights, action_weights, valid_windows


def compare_chunk_effect(name, values, weights, saved):
    total = weights.sum()
    if total <= 0:
        return
    # Avoid materializing 50 overlapping copies of every action.
    mean = np.average(values, axis=0, weights=weights)
    var = np.average((values.astype(np.float64) - mean) ** 2, axis=0, weights=weights)
    std = np.sqrt(np.maximum(0, var))
    print(f"\n=== {name}: exact raw frames vs complete-action-window weighting ===")
    print(f"Effective vectors counted in H-step windows: {int(total):,}")
    for field, weighted in (("mean", mean), ("std", std)):
        original = np.asarray(getattr(saved, field))
        error = np.abs(original - weighted)
        d = int(np.argmax(error))
        print(f"{field}: max |delta|={error[d]:.9g} at dim {d}; raw={original[d]:.9g}, window-weighted={weighted[d]:.9g}")
        print("      window-weighted:", np.array2string(weighted, precision=6, separator=", "))
    print("Note: quantiles for the window-weighted distribution are not computed here.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/data1/dataset/lerobot/libero_plus"))
    p.add_argument("--stats-dir", type=Path, default=Path("assets/pi05_libero_plus/lerobot/libero_plus"))
    p.add_argument("--action-horizon", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32_768)
    args = p.parse_args()
    saved = normalize.load(args.stats_dir)
    adapter = LeRobotV3Adapter(root=args.root, action_horizon=1, decode_images=False, camera_keys=())
    try:
        epids = sorted(adapter.episodes)
        states = np.concatenate([adapter._states[epid] for epid in epids], axis=0)
        actions = np.concatenate([adapter._actions[epid] for epid in epids], axis=0)
        lengths = [int(adapter.episodes[epid]["length"]) for epid in epids]
        assert states.shape[0] == actions.shape[0] == sum(lengths)
        print(f"Dataset: {args.root}, episodes={len(epids):,}, original frames={len(states):,}")
        print(f"Saved stats: {args.stats_dir / 'norm_stats.json'}")
        check_saved_vs_official("state", states, saved["state"], args.batch_size)
        check_saved_vs_official("actions", actions, saved["actions"], args.batch_size)
        if args.action_horizon > 1:
            state_w, action_w, nw = window_weights(lengths, args.action_horizon)
            print(f"\nComplete action windows: {nw:,} (H={args.action_horizon})")
            compare_chunk_effect("state", states, state_w, saved["state"])
            compare_chunk_effect("actions", actions, action_w, saved["actions"])
        print("\nPASS: comparison finished (read-only; no stats overwritten)")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()