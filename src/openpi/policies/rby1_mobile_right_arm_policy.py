"""RBY1 right-arm mobile-manipulation transforms for OpenPI.

Dataset layout
---------------
State 18D:
    right EE 6D + right gripper 1D
    + wheel velocity 4D + right arm joints 7D

Action 10D:
    right absolute EE target 6D
    + right gripper 1D + base velocity 3D
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms


RBY1_WHOLE_BODY_STATE_DIM = 18
RBY1_WHOLE_BODY_ACTION_DIM = 10


def _parse_image(image) -> np.ndarray:
    """LeRobot/OpenPI 이미지를 HWC uint8 형식으로 통일한다."""
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(
            f"Expected a 3D image, got shape {image.shape}"
        )

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(
            image * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)

    elif image.dtype != np.uint8:
        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    if image.shape[0] == 3:
        image = einops.rearrange(
            image,
            "c h w -> h w c",
        )

    if image.shape[-1] != 3:
        raise ValueError(
            f"Expected an RGB image, got shape {image.shape}"
        )

    return image


def make_rby1_whole_body_example() -> dict:
    """정책 생성 및 추론 연결 확인용 observation을 반환한다."""
    return {
        "observation/front_image": np.random.randint(
            256,
            size=(480, 640, 3),
            dtype=np.uint8,
        ),
        "observation/right_image": np.random.randint(
            256,
            size=(640, 480, 3),
            dtype=np.uint8,
        ),
        "observation/state": np.random.rand(
            RBY1_WHOLE_BODY_STATE_DIM
        ).astype(np.float32),
        "prompt": "Approach the door and open it",
    }


@dataclasses.dataclass(frozen=True)
class RBY1WholeBodyInputs(transforms.DataTransformFn):
    """RBY1 오른팔+모바일베이스 입력을 OpenPI 형식으로 변환한다.

    State 18D:
      0:6    right EE [x, y, z, wx, wy, wz]
      6      right gripper
      7:11   wheel velocity [fr, fl, rr, rl]
      11:18  right arm joints [0:7]

    Action 10D:
      0:6    right absolute EE target
      6      right gripper
      7:10   base velocity [x.vel, y.vel, theta.vel]

    action은 delta로 변환하지 않는다.
    base velocity는 wheel velocity에 대한 delta가 아니라
    로봇에 전달할 절대 속도 명령이다.
    """

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(
            data["observation/front_image"]
        )
        right_image = _parse_image(
            data["observation/right_image"]
        )
        state = np.asarray(
            data["observation/state"],
            dtype=np.float32,
        )

        if state.shape[-1] != RBY1_WHOLE_BODY_STATE_DIM:
            raise ValueError(
                f"Expected {RBY1_WHOLE_BODY_STATE_DIM}D "
                f"state, got shape {state.shape}"
            )

        # π0.5의 세 이미지 슬롯 중 left wrist만 없다.
        empty_left_image = np.zeros_like(base_image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": empty_left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = np.asarray(
                data["actions"],
                dtype=np.float32,
            )

            if (
                actions.shape[-1]
                != RBY1_WHOLE_BODY_ACTION_DIM
            ):
                raise ValueError(
                    f"Expected "
                    f"{RBY1_WHOLE_BODY_ACTION_DIM}D "
                    f"actions, got shape {actions.shape}"
                )

            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RBY1WholeBodyOutputs(transforms.DataTransformFn):
    """패딩된 π0.5 출력에서 실제 10D action만 추출한다."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(
            data["actions"],
            dtype=np.float32,
        )

        return {
            "actions": actions[
                ...,
                :RBY1_WHOLE_BODY_ACTION_DIM,
            ]
        }