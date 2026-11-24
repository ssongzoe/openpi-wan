"""RB-Y1 whole-body (24D state / 31D action) OpenPI data transforms.

Dataset: ``rainbowrobotics/wb_test_20261118`` (LeRobot v3.0).

State (24D):
    [0:6]   torso joint positions
    [6:13]  right-arm joint positions
    [13:20] left-arm joint positions
    [20:22] head joint positions
    [22:24] right / left gripper positions

Action (31D):
    [0:6]   torso absolute EE target (x, y, z, wx, wy, wz)
    [6:12]  right-arm absolute EE target (x, y, z, wx, wy, wz)
    [12:18] left-arm absolute EE target (x, y, z, wx, wy, wz)
    [18:21] right-arm null-space commands (3D)
    [21:24] left-arm null-space commands (3D)
    [24:26] head position targets (2D)
    [26:28] right / left gripper targets (2D)
    [28:31] base velocity commands (x, y, theta)

Actions are passed through unchanged: no delta conversion and no joint/EE
conversion. The six null-space action dimensions are *real commands*.

The model's existing ``PadStatesAndActions(32)`` transform adds one zero
column to the action and pads state to 32D *after* normalization/tokenization.
This module intentionally does not perform that padding itself.

Accepted source keys:
    OpenPI repacked: ``observation/front_image``, ``observation/left_image``,
    ``observation/right_image``, ``observation/state``, ``actions``.

    Direct adapter: ``observation.images.front``, ``observation.images.left``,
    ``observation.images.right``, ``observation.state``, ``action``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms


RBY1_WHOLE_BODY_STATE_DIM = 24
RBY1_WHOLE_BODY_ACTION_DIM = 31
RBY1_WHOLE_BODY_MODEL_ACTION_DIM = 32

DEFAULT_BOBBIN_PROMPT = (
    "Pick up the six bobbins from the cart one by one in a fixed sequence, "
    "and place each bobbin into its designated slot on the rack."
)


def _read(data: dict, *keys: str):
    """Get the first matching key without silently substituting a missing view."""
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing required key; expected one of {keys}")


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Convert a single RGB camera image to HWC uint8 (no resize)."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image (HWC or CHW), got {image.shape}")

    # LeRobot video adapter returns HWC RGB. A torch-style CHW is also accepted.
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 RGB channels, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("Image contains NaN or Inf")
        # Floating RGB may be [0,1] or [0,255]; the adapter uses uint8.
        if image.min() >= 0 and image.max() <= 1:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def make_rby1_whole_body_31action_example() -> dict:
    """Build an example *repacked* observation for policy smoke tests."""
    return {
        "observation/front_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation/left_image": np.zeros((640, 480, 3), dtype=np.uint8),
        "observation/right_image": np.zeros((640, 480, 3), dtype=np.uint8),
        "observation/state": np.zeros(RBY1_WHOLE_BODY_STATE_DIM, dtype=np.float32),
        "prompt": DEFAULT_BOBBIN_PROMPT,
    }


@dataclasses.dataclass(frozen=True)
class RBY1WholeBody31ActionInputs(transforms.DataTransformFn):
    """Convert the 3-camera 24D/31D RB-Y1 sample into OpenPI fields.

    ``prompt_override`` replaces the old dataset task text without modifying
    the LeRobot metadata. If ``None``, the sample's prompt is passed through.
    """

    prompt_override: str | None = None

    def __call__(self, data: dict) -> dict:
        front_image = _parse_image(
            _read(data, "observation/front_image", "observation.images.front")
        )
        left_image = _parse_image(
            _read(data, "observation/left_image", "observation.images.left")
        )
        right_image = _parse_image(
            _read(data, "observation/right_image", "observation.images.right")
        )

        state = np.asarray(
            _read(data, "observation/state", "observation.state"),
            dtype=np.float32,
        )
        if state.ndim != 1 or state.shape[0] != RBY1_WHOLE_BODY_STATE_DIM:
            raise ValueError(
                f"Expected one {RBY1_WHOLE_BODY_STATE_DIM}D state, got {state.shape}"
            )

        result = {
            "state": state,
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Direct adapter uses "action"; OpenPI RepackTransform uses "actions".
        action_key = "actions" if "actions" in data else "action"
        if action_key in data:
            actions = np.asarray(data[action_key], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != RBY1_WHOLE_BODY_ACTION_DIM:
                raise ValueError(
                    "Expected action chunk shaped (H, "
                    f"{RBY1_WHOLE_BODY_ACTION_DIM}), got {actions.shape}"
                )
            result["actions"] = actions

            if "action_valid_mask" in data:
                mask = np.asarray(data["action_valid_mask"], dtype=bool)
                if mask.shape != actions.shape[:-1]:
                    raise ValueError(
                        "action_valid_mask shape mismatch: "
                        f"{mask.shape} vs {actions.shape[:-1]}"
                    )
                result["action_valid_mask"] = mask

        if self.prompt_override is not None:
            result["prompt"] = self.prompt_override
        elif "prompt" in data:
            result["prompt"] = data["prompt"]

        return result


@dataclasses.dataclass(frozen=True)
class RBY1WholeBody31ActionOutputs(transforms.DataTransformFn):
    """Strip π0.5's padding and return exactly the 31 command dimensions."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim < 2 or actions.shape[-1] < RBY1_WHOLE_BODY_ACTION_DIM:
            raise ValueError(
                "Expected actions with last dimension >= "
                f"{RBY1_WHOLE_BODY_ACTION_DIM}, got {actions.shape}"
            )
        return {"actions": actions[..., :RBY1_WHOLE_BODY_ACTION_DIM]}