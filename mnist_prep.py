"""
The MNIST input convention, in one place.

This module exists because the same three lines of preprocessing were
previously written out separately in the training script and in each of the
two inference scripts, and they drifted apart. A model is only as correct as
the agreement between how it was trained and how it is called, and that
agreement is impossible to maintain when it lives in three files.

MNIST's convention, which every digit fed to the model must match:

1. **28 x 28 pixels**, single channel.
2. **White digit on a black background.** This is the one that bites. A photo
   or a scan of handwriting is dark ink on light paper, the opposite, and a
   model handed an un-inverted image will confidently return nonsense rather
   than fail loudly.
3. **Scaled to [0, 1]**, not left at [0, 255]. Feeding raw byte values to a
   network trained on normalised input pushes every activation far outside
   the range it saw in training.
4. **Shape (batch, 28, 28, 1)** for a conv model.

`prepare()` enforces all four.
"""

from __future__ import annotations

import numpy as np

IMAGE_SIZE = 28


def prepare(image, invert=False):
    """Bring a single grayscale image into the model's input convention.

    Parameters
    ----------
    image : ndarray, shape (H, W), dtype uint8 or float
        One grayscale image. Colour images must be converted before this.
    invert : bool
        True when the source is dark-ink-on-light-paper (a scan, a photo, a
        webcam frame). False when the source is already white-on-black, which
        is what a thresholded binary image and MNIST itself both are.

    Returns
    -------
    ndarray, shape (1, 28, 28, 1), float32 in [0, 1]
        Ready to hand straight to `model.predict`.
    """
    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a single-channel 2-D image, got shape {image.shape}. "
            f"Convert colour to grayscale first."
        )

    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        # Imported lazily so the training script, which never resizes, does
        # not need OpenCV installed.
        import cv2

        image = cv2.resize(
            image.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
        )

    image = image.astype(np.float32)

    if invert:
        image = 255.0 - image

    image /= 255.0

    return image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)


def prepare_batch(images):
    """Scale and reshape an already-correct-polarity batch, e.g. MNIST itself.

    Parameters
    ----------
    images : ndarray, shape (n, 28, 28)

    Returns
    -------
    ndarray, shape (n, 28, 28, 1), float32 in [0, 1]
    """
    images = np.asarray(images, dtype=np.float32) / 255.0
    return images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
