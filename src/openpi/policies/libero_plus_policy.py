"""OpenPI data transforms for the LeRobot v3.0 LIBERO-Plus dataset.

Dataset: ``lerobot/libero_plus``

    observation.images.front: RGB image (256, 256, 3)
    observation.images.wrist: RGB image (256, 256, 3)
    observation.state:        (8,) float32
    action:                   (7,) float32 per frame
    prompt:                   language instruction

The LeRobotV3Adapter returns an (H, 7) action chunk. LIBERO actions
are already relative control commands, so they are *not* delta-converted.

This transform emits the canonical OpenPI field names. OpenPI's existing
ModelTransformFactory subsequently resizes images, tokenizes the prompt,
normalizes the raw 8D/7D values (normalization occurs before model padding),
and pads state/actions to the model's 32D action space.

Accepted keys:
    Raw LeRobotV3Adapter:
        observation.images.front, observation.images.wrist,
        observation.state, action

    OpenPI-style repacked:
        observation/image, observation/wrist_image,
        observation/state, actions

Third OpenPI camera is deliberately absent from LIBERO-Plus: it is represented
as an all-zero image with image_mask=False for Pi0/Pi05. This matches the
upstream OpenPI LIBERO convention.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


LIBERO_PLUS_STATE_DIM = 8
LIBERO_PLUS_ACTION_DIM = 7
LIBERO_PLUS_MODEL_ACTION_DIM = 32


def _get(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing required LIBERO-Plus field; tried keys: {keys}")


def _rgb_uint8(value, *, camera_name: str) -> np.ndarray:
    """Return one HWC RGB uint8 image; accepts HWC or CHW float/uint8."""
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{camera_name} must be one HWC/CHW image; got {image.shape}")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"{camera_name} must have three RGB channels; got {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError(f"{camera_name} contains nonfinite values")
        if image.min() < 0 or image.max() > 255:
            raise ValueError(f"{camera_name} float RGB should be in [0,1] or [0,255]")
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.astype(np.uint8)
    elif image.dtype != np.uint8:
        raise TypeError(f"{camera_name} must be uint8 or floating point RGB; got {image.dtype}")

    return image


def make_libero_plus_example(*, action_horizon: int = 50) -> dict:
    """Create a raw LeRobot v3.0 sample for manual/data transform testing."""
    return {
        "observation.images.front": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.images.wrist": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation.state": np.zeros((LIBERO_PLUS_STATE_DIM,), dtype=np.float32),
        "action": np.zeros((action_horizon, LIBERO_PLUS_ACTION_DIM), dtype=np.float32),
        "action_valid_mask": np.ones((action_horizon,), dtype=np.bool_),
        "prompt": "pick up the object and place it in the target location",
    }


@dataclasses.dataclass(frozen=True)
class LiberoPlusInputs(transforms.DataTransformFn):
    """LeRobot v3.0 LIBERO-Plus sample -> OpenPI sample.

    ``model_type`` is used only to match upstream LIBERO's dummy image-mask
    behavior (PI0_FAST requires True; PI0/PI05 use False).

    Do not pass a ``prompt_override`` for a multi-task LIBERO-Plus run unless
    you deliberately intend to replace every task's language instruction.
    """

    model_type: _model.ModelType
    prompt_override: str | None = None

    def __call__(self, data: dict) -> dict:
        front = _rgb_uint8(
            _get(data, "observation/image", "observation.images.front", "observation/front_image"),
            camera_name="front",
        )
        wrist = _rgb_uint8(
            _get(data, "observation/wrist_image", "observation.images.wrist", "observation/wrist_image"),
            camera_name="wrist",
        )
        state = np.asarray(
            _get(data, "observation/state", "observation.state"), dtype=np.float32
        )
        if state.shape != (LIBERO_PLUS_STATE_DIM,):
            raise ValueError(
                f"LIBERO-Plus state must be ({LIBERO_PLUS_STATE_DIM},), got {state.shape}"
            )
        if not np.isfinite(state).all():
            raise ValueError("LIBERO-Plus state contains NaN/Inf")

        result = {
            "state": state,
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(front),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.bool_(self.model_type == _model.ModelType.PI0_FAST),
            },
        }

        if "actions" in data or "action" in data:
            actions = np.asarray(
                _get(data, "actions", "action"), dtype=np.float32
            )
            if actions.ndim != 2 or actions.shape[1] != LIBERO_PLUS_ACTION_DIM:
                raise ValueError(
                    f"LIBERO-Plus action chunk must be (H, {LIBERO_PLUS_ACTION_DIM}), "
                    f"got {actions.shape}"
                )
            if not np.isfinite(actions).all():
                raise ValueError("LIBERO-Plus actions contain NaN/Inf")
            # No delta/absolute conversion: keep all 7 raw action dimensions.
            result["actions"] = actions

            if "action_valid_mask" in data:
                mask = np.asarray(data["action_valid_mask"], dtype=np.bool_)
                if mask.shape != actions.shape[:-1]:
                    raise ValueError(
                        "action_valid_mask must match action horizon; "
                        f"got {mask.shape} vs {actions.shape[:-1]}"
                    )
                result["action_valid_mask"] = mask

        if self.prompt_override is not None:
            result["prompt"] = self.prompt_override
        elif "prompt" in data:
            result["prompt"] = data["prompt"]

        return result


@dataclasses.dataclass(frozen=True)
class LiberoPlusOutputs(transforms.DataTransformFn):
    """OpenPI padded action -> the 7D LIBERO control action sequence."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim < 1 or actions.shape[-1] < LIBERO_PLUS_ACTION_DIM:
            raise ValueError(
                f"Expected last action dimension >= {LIBERO_PLUS_ACTION_DIM}, "
                f"got {actions.shape}"
            )
        return {"actions": actions[..., :LIBERO_PLUS_ACTION_DIM]}


def _smoke_test() -> None:
    from openpi.models.model import ModelType

    example = make_libero_plus_example(action_horizon=50)
    example["action"][:, 6] = 1.0
    policy = LiberoPlusInputs(model_type=ModelType.PI05)
    out = policy(example)
    assert out["state"].shape == (8,)
    assert out["actions"].shape == (50, 7)
    assert np.array_equal(out["actions"], example["action"])
    assert out["action_valid_mask"].shape == (50,)
    assert np.all(out["action_valid_mask"])
    assert out["image"]["base_0_rgb"].shape == (256, 256, 3)
    assert out["image"]["left_wrist_0_rgb"].shape == (256, 256, 3)
    assert out["image"]["right_wrist_0_rgb"].shape == (256, 256, 3)
    assert not bool(out["image_mask"]["right_wrist_0_rgb"])
    assert out["prompt"] == example["prompt"]

    # Check that source keys used by the official OpenPI LIBERO config work.
    repacked = {
        "observation/image": np.moveaxis(example["observation.images.front"].astype(np.float32) / 255.0, -1, 0),
        "observation/wrist_image": example["observation.images.wrist"],
        "observation/state": example["observation.state"],
        "actions": example["action"],
        "action_valid_mask": example["action_valid_mask"],
        "prompt": example["prompt"],
    }
    repacked_out = policy(repacked)
    assert np.array_equal(repacked_out["actions"], example["action"])

    padded = np.zeros((50, LIBERO_PLUS_MODEL_ACTION_DIM), dtype=np.float32)
    padded[:, :LIBERO_PLUS_ACTION_DIM] = example["action"]
    restored = LiberoPlusOutputs()({"actions": padded})["actions"]
    assert restored.shape == (50, 7)
    assert np.array_equal(restored, example["action"])

    print("PASS: LIBERO-Plus 2 cameras / state=8D / action=7D / mask / output slice")
    print("Model padding: state 8D -> 32D, action 7D -> 32D (done by OpenPI ModelTransforms)")


if __name__ == "__main__":
    _smoke_test()