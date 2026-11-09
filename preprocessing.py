"""Reusable image recovery methods used by the robustness benchmark and CLI."""
from __future__ import annotations

import cv2
import numpy as np

RECOVERY_METHODS = ("auto", "median", "nlmeans", "clahe", "bilateral")
NOISE_FAMILIES = ("gaussian", "salt_pepper", "fog", "low_light")


def recover_image(image: np.ndarray, source_family: str, method: str = "auto") -> np.ndarray:
    """Apply one recovery method to a BGR uint8 image and return a BGR uint8 image."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a decoded three-channel BGR image")
    if image.dtype != np.uint8:
        raise ValueError("image dtype must be uint8")
    if source_family not in NOISE_FAMILIES:
        raise ValueError(f"source_family must be one of {NOISE_FAMILIES}")
    if method not in RECOVERY_METHODS:
        raise ValueError(f"method must be one of {RECOVERY_METHODS}")

    selected = method
    if method == "auto":
        selected = {"salt_pepper": "median", "gaussian": "nlmeans",
                    "low_light": "clahe", "fog": "clahe"}[source_family]
    if selected == "median":
        return cv2.medianBlur(image, 3)
    if selected == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
    if selected == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
        return cv2.cvtColor(cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    if selected == "bilateral":
        return cv2.bilateralFilter(image, 7, 50, 50)
    raise AssertionError(f"unhandled recovery method {selected}")
