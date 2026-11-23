import dataclasses

import einops
import numpy as np

from openpi import transforms


ACTION_DIM = 16


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RBY1Inputs(transforms.DataTransformFn):
    """Inputs for the RBY1 policy.

    Expected action/state layout:
    - dims 0:7: right arm joints
    - dims 7:14: left arm joints
    - dim 14: right gripper
    - dim 15: left gripper
    """

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/front_image"])
        right_image = _parse_image(data["observation/right_image"])
        left_image = _parse_image(data["observation/left_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RBY1Outputs(transforms.DataTransformFn):
    """Outputs for the RBY1 policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
    



RBY1_EE_STATE_DIM = 26
RBY1_EE_ACTION_DIM = 12


def make_rby1_ee_example() -> dict:
    """Single-camera RBY1 EE inference example."""
    return {
        "observation/front_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(
            RBY1_EE_STATE_DIM
        ).astype(np.float32),
        "prompt": "Tilt the box using both arms.",
    }


@dataclasses.dataclass(frozen=True)
class RBY1EEInputs(transforms.DataTransformFn):
    """Inputs for single-camera absolute EE control.

    State:
      0:14   bilateral arm joints
      14:20  right EE [x, y, z, wx, wy, wz]
      20:26  left EE  [x, y, z, wx, wy, wz]

    Action:
      0:6    right absolute EE target
      6:12   left absolute EE target
    """

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/front_image"])
        state = np.asarray(
            data["observation/state"],
            dtype=np.float32,
        )

        if state.shape[-1] != RBY1_EE_STATE_DIM:
            raise ValueError(
                f"Expected {RBY1_EE_STATE_DIM}D state, "
                f"got shape {state.shape}"
            )

        # π0.5는 표준적으로 이미지 슬롯 3개를 사용한다.
        # 이 데이터셋은 front 카메라만 있으므로 wrist 이미지는
        # zero image로 채우고 mask=False로 비활성화한다.
        empty_wrist_image = np.zeros_like(base_image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": empty_wrist_image,
                "right_wrist_0_rgb": empty_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(
                data["actions"],
                dtype=np.float32,
            )

            if actions.shape[-1] != RBY1_EE_ACTION_DIM:
                raise ValueError(
                    f"Expected {RBY1_EE_ACTION_DIM}D actions, "
                    f"got shape {actions.shape}"
                )

            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RBY1EEOutputs(transforms.DataTransformFn):
    """Extract 12D EE actions from padded π0.5 outputs."""

    def __call__(self, data: dict) -> dict:
        return {
            "actions": np.asarray(data["actions"])[
                ..., :RBY1_EE_ACTION_DIM
            ]
        }