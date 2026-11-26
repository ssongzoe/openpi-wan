"""Pre-encode LeRobot v3 episode videos using frozen Wan2.2 TI2V-5B VAE.

Stores aligned, normalized Wan video latents for the training-only
FutureVideoPredictor, NOT images or action labels. Wan2.2 VAE is causal:
for input frames (0, 1, ..., 4*k), it produces k+1 latent frames,
anchored to source frame indices (0, 4, ..., 4*k).

Each episode is encoded in overlapping clips of 4*k+1 frames, with ONE
shared boundary frame; the duplicate boundary latent is dropped. This
resets the VAE's causal state between clips (a practical memory tradeoff).
No clip ever crosses an episode boundary.

The stored ``latents`` are [C=48,T,H,W], already standardized by the Wan
VAE config's per-channel latents_mean/latents_std, compatible with unit
Gaussian flow-matching noise. For the predictor: stack windows as
[B,48,T,H,W], set future_valid_mask[:,0]=False.

Default camera: front only. Camera views must NOT be concatenated on the
latent channel axis, since FutureVideoPredictor currently expects C=48.

Requirements: Python >=3.10, numpy, pyarrow, av, Pillow, torch, diffusers,
transformers, huggingface_hub. GPU strongly recommended.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterator

import numpy as np

from openpi_wan.adapters.lerobot_v3_adapter import LeRobotV3Adapter

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
SCHEMA_VERSION = 1
TEMPORAL_STRIDE = 4
LATENT_CHANNELS = 48
SPATIAL_STRIDE = 16


def _resize_rgb(frame: np.ndarray, size: int, mode: str) -> np.ndarray:
    """Resize one RGB frame; use same preprocessing for every camera/episode."""
    from PIL import Image

    im = Image.fromarray(frame, mode="RGB")
    if mode == "stretch":
        im = im.resize((size, size), resample=Image.Resampling.BICUBIC)
    elif mode == "letterbox":
        scale = min(size / im.width, size / im.height)
        new_w = max(1, round(im.width * scale))
        new_h = max(1, round(im.height * scale))
        content = im.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(content, ((size - new_w) // 2, (size - new_h) // 2))
        im = canvas
    else:
        raise ValueError(f"Unknown resize mode: {mode}")
    return np.asarray(im, dtype=np.uint8)


def _frame_pts_seconds(frame, stream) -> float | None:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None:
        return float(frame.pts * stream.time_base)
    return None


def iter_episode_rgb(
    dataset: LeRobotV3Adapter, ep_id: int, video_key: str,
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream RGB frames in frame-index order with v3 shard timestamp alignment.

    Unlike ``dataset[i]`` this seeks once and decodes sequentially. Every
    output is checked against the *original Parquet timestamp* plus the
    metadata episode's ``from_timestamp`` offset in the shared MP4 shard.
    """
    try:
        import av
    except ImportError as exc:
        raise ImportError("Install PyAV: python -m pip install av") from exc

    ep = dataset.episodes[ep_id]
    times = np.asarray(dataset._timestamps[ep_id], dtype=np.float64)
    base = float(ep[f"videos/{video_key}/from_timestamp"])
    end = float(ep[f"videos/{video_key}/to_timestamp"])
    targets = base + times
    tol = 1.01 / dataset.fps
    if targets[-1] > end + tol:
        raise ValueError(f"Episode {ep_id} exceeds {video_key} video metadata range")

    path = dataset._video_path(ep_id, video_key)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        seek_s = max(0.0, float(targets[0]) - max(0.5, 4.0 / dataset.fps))
        container.seek(int(seek_s / float(stream.time_base)), stream=stream, backward=True)

        previous_time = None
        previous_frame = None
        target_i = 0
        for frame in container.decode(stream):
            now = _frame_pts_seconds(frame, stream)
            if now is None:
                continue
            # Handle target frames between the previous/current decoded frames.
            while target_i < len(targets) and targets[target_i] <= now:
                target = float(targets[target_i])
                chosen, err = frame, abs(now - target)
                if previous_frame is not None:
                    old_err = abs(float(previous_time) - target)
                    if old_err < err:
                        chosen, err = previous_frame, old_err
                if err > tol:
                    raise RuntimeError(
                        f"Video/Parquet alignment failed: episode={ep_id}, camera={video_key}, "
                        f"frame={target_i}, target={target:.6f}s, error={err:.6f}s "
                        f"(limit={tol:.6f}s)."
                    )
                yield target_i, chosen.to_ndarray(format="rgb24")
                target_i += 1
            if target_i == len(targets):
                return
            previous_frame, previous_time = frame, now
            if now > end + tol:
                break

        # Allow the final target to match the final decoded frame, if close.
        while target_i < len(targets) and previous_frame is not None:
            err = abs(float(previous_time) - float(targets[target_i]))
            if err > tol:
                break
            yield target_i, previous_frame.to_ndarray(format="rgb24")
            target_i += 1
        if target_i != len(targets):
            raise RuntimeError(
                f"Video ended early / timestamps misaligned: episode={ep_id}, "
                f"camera={video_key}, matched={target_i}/{len(targets)}, path={path}"
            )


def _chunk_video(
    frames: Iterator[tuple[int, np.ndarray]], clip_frames: int,
) -> Iterator[tuple[int, list[np.ndarray]]]:
    """Yield contiguous (4*k+1)-frame clips, overlapping at one anchor frame.

    Leftover fewer than 5 frames cannot produce a new *future* latent.
    They are intentionally discarded (not padded into fake future labels).
    """
    buffer: list[np.ndarray] = []
    start = 0
    expected = 0
    for idx, frame in frames:
        if idx != expected:
            raise RuntimeError(f"Expected sequential frame index {expected}, received {idx}")
        expected += 1
        buffer.append(frame)
        if len(buffer) == clip_frames:
            yield start, buffer
            start += clip_frames - 1
            buffer = [buffer[-1]]

    n_valid = 1 + TEMPORAL_STRIDE * ((len(buffer) - 1) // TEMPORAL_STRIDE)
    if n_valid >= 5:
        yield start, buffer[:n_valid]
    elif expected == 1 and buffer:
        # Single-frame episode: the first latent is still a valid current block.
        yield 0, buffer
    elif expected > 1 and start == 0 and len(buffer) >= 1:
        # Very short 2-4 frame episode: encode the first/current latent only.
        yield 0, buffer[:1]


def _load_frozen_vae(model_id: str, device: str, dtype_name: str):
    try:
        import torch
        from diffusers import AutoencoderKLWan
    except ImportError as exc:
        raise ImportError(
            "Wan VAE requires torch and recent diffusers: "
            "python -m pip install -U torch diffusers transformers accelerate safetensors"
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({device}) but torch.cuda.is_available() is False")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    print(f"Loading frozen VAE only: {model_id} (subfolder=vae, device={device}, dtype={dtype_name})", flush=True)
    # Do NOT load the 5B diffusion transformer / text encoder / pipeline.
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.to(device=device, dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)
    cfg = vae.config
    assert int(cfg.z_dim) == LATENT_CHANNELS, f"Expected z_dim=48, got {cfg.z_dim}"
    assert int(cfg.scale_factor_spatial) == SPATIAL_STRIDE
    assert int(cfg.scale_factor_temporal) == TEMPORAL_STRIDE
    means = torch.as_tensor(cfg.latents_mean, dtype=torch.float32, device=device).reshape(1, 48, 1, 1, 1)
    stds = torch.as_tensor(cfg.latents_std, dtype=torch.float32, device=device).reshape(1, 48, 1, 1, 1)
    if bool((stds <= 0).any()):
        raise ValueError("Wan VAE latents_std contains non-positive entries")
    return vae, means, stds


def _encode_clip(
    vae, means, stds, rgb_frames: list[np.ndarray], *, device: str, latent_dtype: str,
) -> np.ndarray:
    """Encode a clip with posterior MODE and Wan model-specific standardization."""
    import torch

    n = len(rgb_frames)
    if (n - 1) % 4 != 0:
        raise ValueError(f"Wan clips must have 4*k+1 frames; got {n}")
    frames = np.stack(rgb_frames, axis=0)  # uint8 [F,H,W,3]
    # RGB pixels [0,255] -> [-1,1], shape [1,3,F,H,W].
    x = torch.from_numpy(frames).to(device=device)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).to(dtype=vae.dtype)
    x = x / 127.5 - 1.0
    with torch.inference_mode():
        # Deterministic posterior mean / mode, not a random latent sample.
        latent = vae.encode(x).latent_dist.mode().float()
        latent = (latent - means) / stds
    if latent.ndim != 5 or latent.shape[:3] != (1, 48, 1 + (n - 1) // 4):
        raise RuntimeError(f"Unexpected Wan VAE latent shape for {n} frames: {tuple(latent.shape)}")
    latent_np = latent[0].to(device="cpu", dtype=torch.float32).numpy()
    if not np.isfinite(latent_np).all():
        raise ValueError("Non-finite Wan latent values detected")
    return latent_np.astype(np.float16 if latent_dtype == "float16" else np.float32)


def _write_npz(path: Path, arrays: dict[str, np.ndarray], *, compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".partial.npz")
    try:
        with temp.open("wb") as f:
            if compress:
                np.savez_compressed(f, **arrays)
            else:
                np.savez(f, **arrays)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _check_existing(path: Path, *, ep_id: int, n_source_frames: int) -> bool:
    """True iff a previously saved episode has the expected aligned frame set."""
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as data:
        frames = data["frame_indices"]
        z = data["latents"]
        expected = np.arange(0, n_source_frames, 4, dtype=np.int32)
        if z.ndim != 4 or z.shape[0] != 48 or z.shape[1] != len(frames):
            raise ValueError(f"Bad saved latent shape: {path}: {z.shape}")
        if not np.array_equal(frames, expected):
            raise ValueError(f"Saved frame alignment mismatch: {path}")
        if int(data["episode_index"].item()) != ep_id:
            raise ValueError(f"Wrong episode_index in {path}")
    return True


def encode_episode(
    dataset: LeRobotV3Adapter,
    *, ep_id: int, camera: str, size: int, resize_mode: str, clip_frames: int,
    vae, means, stds, device: str, latent_dtype: str, output_path: Path, compress: bool,
) -> dict:
    key = f"observation.images.{camera}"
    n = int(dataset.episodes[ep_id]["length"])
    timestamps = dataset._timestamps[ep_id]
    source_frames = iter_episode_rgb(dataset, ep_id, key)
    resized = ((i, _resize_rgb(frame, size, resize_mode)) for i, frame in source_frames)

    pieces = []
    indices = []
    for clip_i, (clip_start, clip) in enumerate(_chunk_video(resized, clip_frames)):
        latent = _encode_clip(vae, means, stds, clip, device=device, latent_dtype=latent_dtype)
        # Drop repeated anchor latent at clip boundaries; no duplicate labels.
        start_z = 0 if clip_i == 0 else 1
        pieces.append(latent[:, start_z:])
        indices.extend(range(clip_start + 4 * start_z, clip_start + len(clip), 4))

    if not pieces:
        raise RuntimeError(f"No VAE clips for episode={ep_id}, camera={camera}")
    z = np.concatenate(pieces, axis=1)
    frame_ids = np.asarray(indices, dtype=np.int32)
    expected = np.arange(0, n, 4, dtype=np.int32)
    if not np.array_equal(frame_ids, expected):
        raise AssertionError(
            f"Episode latent alignment mismatch: ep={ep_id} "
            f"expected {expected.shape} indices, got {frame_ids.shape}: "
            f"tail={frame_ids[-5:]} expected_tail={expected[-5:]}"
        )
    if z.shape != (48, len(frame_ids), size // 16, size // 16):
        raise AssertionError(f"Unexpected final latent shape: {z.shape}")
    _write_npz(
        output_path,
        {
            "latents": z,
            "frame_indices": frame_ids,
            "timestamps_sec": timestamps[frame_ids].astype(np.float32),
            "episode_index": np.asarray(ep_id, dtype=np.int32),
            "source_num_frames": np.asarray(n, dtype=np.int32),
        },
        compress=compress,
    )
    return {
        "episode_index": ep_id,
        "camera": camera,
        "frames": n,
        "latent_frames": len(frame_ids),
        "latent_shape": list(z.shape),
        "path": str(output_path),
        "first_source_frame": int(frame_ids[0]),
        "last_source_frame": int(frame_ids[-1]),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="LeRobot v3 dataset root")
    p.add_argument("--output", type=Path, default=Path("./assets/wan22_vae_latents/wb_test_20261118"))
    p.add_argument("--model-id", default=MODEL_ID, help="Diffusers Wan2.2 TI2V-5B model repo/path")
    p.add_argument("--cameras", nargs="+", choices=["front", "left", "right"], default=["front"])
    p.add_argument("--episode-ids", nargs="+", type=int, default=None)
    p.add_argument("--max-episodes", type=int, default=None, help="Process first N episodes (smoke test)")
    p.add_argument("--image-size", type=int, default=256, help="Square Wan VAE input (multiple of 32)")
    p.add_argument("--resize-mode", choices=["letterbox", "stretch"], default="letterbox")
    p.add_argument("--clip-frames", type=int, default=49, help="Clip length 4*k+1, shared boundary frame")
    p.add_argument("--device", default="cuda:0", help="Torch VAE device, respects CUDA_VISIBLE_DEVICES")
    p.add_argument("--vae-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--latent-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--compress", action="store_true", help="ZIP-compress npz files (slower)")
    p.add_argument("--resume", action="store_true", help="Skip previously completed episodes after validation")
    p.add_argument("--overwrite", action="store_true", help="Replace existing matching-config episode files")
    p.add_argument("--dry-run", action="store_true", help="Inspect dataset/plan without loading VAE or writing files")
    args = p.parse_args()
    if args.clip_frames < 5 or (args.clip_frames - 1) % 4:
        p.error("--clip-frames must be 4*k+1 and at least 5 (e.g., 17, 33, 49)")
    if args.image_size < 32 or args.image_size % 32:
        p.error("--image-size must be a multiple of 32 and >= 32 (e.g., 256)")
    if args.max_episodes is not None and args.max_episodes < 1:
        p.error("--max-episodes must be >= 1")
    if args.resume and args.overwrite:
        p.error("--resume and --overwrite cannot be used together")
    if len(set(args.cameras)) != len(args.cameras):
        p.error("--cameras must be unique")
    return args


def main() -> None:
    args = _parse_args()
    dataset = LeRobotV3Adapter(
        args.root, action_horizon=1, decode_images=False, camera_keys=tuple(args.cameras)
    )
    try:
        episode_ids = sorted(dataset.episodes) if args.episode_ids is None else args.episode_ids
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("Duplicate episode ids")
        missing = set(episode_ids) - set(dataset.episodes)
        if missing:
            raise ValueError(f"Unknown episode ids: {sorted(missing)}")
        if args.max_episodes is not None:
            episode_ids = episode_ids[:args.max_episodes]
        total = sum(int(dataset.episodes[e]["length"]) for e in episode_ids)
        latent_count = sum(math.ceil(int(dataset.episodes[e]["length"]) / 4) for e in episode_ids)
        print(f"Dataset: {dataset.root} | FPS={dataset.fps} | episodes={len(episode_ids)}")
        print(f"Cameras: {args.cameras} | source frames/camera: {total:,}")
        print(f"Expected latent frames/camera: {latent_count:,} (indices 0,4,8,... per episode)")
        print(f"Output: {args.output.resolve()} | size={args.image_size} | clip={args.clip_frames}")
        print(f"Expected latent [C,T,H,W] per episode: C=48, H=W={args.image_size // 16}")
        if args.dry_run:
            print("DRY RUN PASS: no video decoded, model downloaded, or file written")
            return

        output = args.output.expanduser().resolve()
        settings = {
            "schema_version": SCHEMA_VERSION,
            "model_id": args.model_id,
            "dataset_root": str(dataset.root),
            "fps": dataset.fps,
            "cameras": sorted(args.cameras),
            "image_size": args.image_size,
            "resize_mode": args.resize_mode,
            "clip_frames": args.clip_frames,
            "vae_dtype": args.vae_dtype,
            "latent_dtype": args.latent_dtype,
            "standardization": "(posterior.mode() - vae.config.latents_mean) / vae.config.latents_std",
            "temporal_stride": TEMPORAL_STRIDE,
            "spatial_stride": SPATIAL_STRIDE,
            "latent_channels": LATENT_CHANNELS,
            "latent_layout": "[C,T,H,W]",
            "frame_anchor": "frame_index=0,4,8,... (episode-local)",
            "video_segment_boundary": "overlap-one-anchor frame; VAE cache reset every clip",
        }
        metadata_path = output / "encoding_config.json"
        if metadata_path.exists():
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            if saved != settings:
                raise ValueError(
                    f"Incompatible existing encoding_config.json: {metadata_path}. "
                    "Use a different --output directory for changed encoder settings."
                )
        else:
            # Refuse to append to an unmarked directory of npz files.
            if output.exists() and list(output.rglob("*.npz")):
                raise ValueError(f"Output directory already contains npz files without config: {output}")
            _write_json(metadata_path, settings)

        remaining = []
        for camera in args.cameras:
            for ep_id in episode_ids:
                save_path = output / camera / f"episode_{ep_id:06d}.npz"
                if save_path.exists():
                    if args.resume and _check_existing(
                        save_path, ep_id=ep_id, n_source_frames=int(dataset.episodes[ep_id]["length"])
                    ):
                        continue
                    if not args.overwrite:
                        raise FileExistsError(f"Existing result: {save_path} (use --resume or --overwrite)")
                remaining.append((camera, ep_id, save_path))
        if not remaining:
            print("PASS: all requested episode/camera files already exist and are aligned")
            return

        vae, means, stds = _load_frozen_vae(args.model_id, args.device, args.vae_dtype)
        for idx, (camera, ep_id, save_path) in enumerate(remaining, start=1):
            result = encode_episode(
                dataset,
                ep_id=ep_id, camera=camera, size=args.image_size,
                resize_mode=args.resize_mode, clip_frames=args.clip_frames,
                vae=vae, means=means, stds=stds, device=args.device,
                latent_dtype=args.latent_dtype, output_path=save_path,
                compress=args.compress,
            )
            print(
                f"[{idx}/{len(remaining)}] camera={camera} episode={ep_id} "
                f"frames={result['frames']} -> latents={result['latent_shape']} "
                f"saved={save_path}", flush=True,
            )
        print("PASS: Wan2.2 VAE encoding and source-frame alignment verified", flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()