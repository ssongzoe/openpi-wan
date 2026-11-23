import dataclasses

import einops
import numpy as np

from openpi import transforms


STATE_DIM = 11
ACTION_DIM = 4


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)

    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    return image


@dataclasses.dataclass(frozen=True)
class RBY1MobileInputs(transforms.DataTransformFn):
    """Inputs for the RBY1 mobile-base-only policy.

    Expected observation/state layout:
    - dim 0: base_x.pos
    - dim 1: base_y.pos
    - dim 2: base_x.vel
    - dim 3: base_y.vel
    - dim 4: base_theta.vel
    - dim 5: wheel_fr.vel
    - dim 6: wheel_fl.vel
    - dim 7: wheel_rr.vel
    - dim 8: wheel_rl.vel
    - dim 9: base_theta.sin
    - dim 10: base_theta.cos

    Expected action layout:
    - dim 0: x.vel
    - dim 1: y.vel
    - dim 2: theta.vel
    - dim 3: endpoint_state
    """

    model_type: str

    def __call__(self, data: dict) -> dict:
        front_image = _parse_image(data["observation/front_image"])
        bottom_image = _parse_image(data["observation/bottom_image"])

        state = np.asarray(data["observation/state"], dtype=np.float32)

        if state.shape[-1] != STATE_DIM:
            raise ValueError(
                f"Expected observation.state dim {STATE_DIM}, got shape {state.shape}"
            )

        # pi0/pi05 계열은 보통 image slot이 base/left_wrist/right_wrist 형태라
        # front -> base_0_rgb
        # bottom -> left_wrist_0_rgb
        # right_wrist_0_rgb -> dummy + mask False
        #
        # inference 쪽에서도 반드시 같은 mapping을 써야 함.
        dummy_image = np.zeros_like(bottom_image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": bottom_image,
                "right_wrist_0_rgb": dummy_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)

            if actions.shape[-1] != ACTION_DIM:
                raise ValueError(
                    f"Expected action dim {ACTION_DIM}, got shape {actions.shape}"
                )

            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RBY1MobileOutputs(transforms.DataTransformFn):
    """Outputs for the RBY1 mobile-base-only policy."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        return {"actions": actions[:, :ACTION_DIM]}