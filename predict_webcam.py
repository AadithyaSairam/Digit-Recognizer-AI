#!/usr/bin/env python3
"""
Live digit recognition from a webcam.

    python predict_webcam.py
    python predict_webcam.py --model digits.keras --min-confidence 0.8

Hold handwritten digits up to the camera. Detected digits are boxed and
labelled. Press `q` to quit.

How it finds digits: adaptive threshold (which handles uneven lighting far
better than a global threshold), then contours, filtered by area and by
solidity, the ratio of a contour's area to its convex hull's. Solidity
rejects the thin, straggly contours that shadows and paper edges produce
while keeping the compact blobs that digits are.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from mnist_prep import prepare

MIN_AREA = 100
MIN_SOLIDITY = 0.5

# MNIST digits sit in a 20x20 box centred in a 28x28 frame. Cropping tight to
# the contour and resizing loses that margin, and the model sees a digit
# noticeably larger than anything in training.
PAD_RATIO = 0.2


def find_digit_contours(thresholded, min_area=MIN_AREA, min_solidity=MIN_SOLIDITY):
    """Contours that plausibly bound a handwritten digit."""
    found = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]

    keep = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= min_area:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity > min_solidity:
            keep.append(contour)

    return keep


def extract_roi(thresholded, x, y, w, h, pad_ratio=PAD_RATIO):
    """Crop a padded, square region around a bounding box.

    Cropped from the *thresholded* image, not the grayscale one. This is the
    subtle part: the threshold used here is `THRESH_BINARY_INV`, so its output
    is already white-digit-on-black, exactly MNIST's convention. Cropping the
    grayscale frame instead yields dark-ink-on-light-paper, the inverse of
    what the model was trained on, and the predictions become noise while
    everything still appears to run.
    """
    pad = int(max(w, h) * pad_ratio)

    # Square the box before padding, so the resize does not stretch the digit.
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    half = side // 2 + pad

    x0 = max(cx - half, 0)
    y0 = max(cy - half, 0)
    x1 = min(cx + half, thresholded.shape[1])
    y1 = min(cy + half, thresholded.shape[0])

    return thresholded[y0:y1, x0:x1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="digits.keras")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(
            f"Model not found: {args.model}\nTrain one first:  python train.py",
            file=sys.stderr,
        )
        return 1

    model = keras.models.load_model(args.model)

    # CAP_DSHOW avoids a multi-second open delay with MSMF on Windows. It is
    # ignored on other platforms.
    capture = cv2.VideoCapture(args.camera + cv2.CAP_DSHOW)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1

    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Gamma < 1 lifts the mid-tones, which helps pencil on white paper.
            gray = np.array(255 * (gray / 255) ** 1.5, dtype=np.uint8)

            thresholded = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )

            for contour in find_digit_contours(thresholded):
                x, y, w, h = cv2.boundingRect(contour)
                roi = extract_roi(thresholded, x, y, w, h)
                if roi.size == 0:
                    continue

                # invert=False: the ROI came from THRESH_BINARY_INV output and
                # is already white-on-black.
                probabilities = model.predict(prepare(roi, invert=False), verbose=0)[0]
                digit = int(np.argmax(probabilities))
                confidence = float(probabilities[digit])

                if confidence > args.min_confidence:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{digit} ({confidence:.2f})",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow("Digit recognition - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
