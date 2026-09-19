#!/usr/bin/env python3
"""
Classify digit images from files.

    python predict_image.py numbers/*.png
    python predict_image.py --model digits.keras --show numbers/num7.png

Images are assumed to be dark ink on a light background — a scan, a photo, or
something drawn in a paint program. They are inverted to MNIST's white-on-black
convention by `mnist_prep.prepare(..., invert=True)`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from mnist_prep import prepare


def load_grayscale(path):
    """Read an image as a single 8-bit channel, or raise with the path."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model", default="digits.keras")
    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="source is already white-on-black, like MNIST itself",
    )
    parser.add_argument("--show", action="store_true", help="display each image")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(
            f"Model not found: {args.model}\nTrain one first:  python train.py",
            file=sys.stderr,
        )
        return 1

    model = keras.models.load_model(args.model)

    failures = 0
    for path in args.images:
        # Deliberately not a bare `except:`. A missing file and a corrupt
        # image should say which they are; anything else is a real bug and
        # should surface with its traceback rather than print "error".
        try:
            image = load_grayscale(path)
        except FileNotFoundError as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            failures += 1
            continue

        batch = prepare(image, invert=not args.no_invert)
        probabilities = model.predict(batch, verbose=0)[0]

        digit = int(np.argmax(probabilities))
        confidence = float(probabilities[digit])
        print(f"{path}: {digit}  (confidence {confidence:.3f})")

        if args.show:
            import matplotlib.pyplot as plt

            plt.imshow(batch[0, :, :, 0], cmap="binary")
            plt.title(f"{path.name} -> {digit} ({confidence:.2f})")
            plt.axis("off")
            plt.show()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
