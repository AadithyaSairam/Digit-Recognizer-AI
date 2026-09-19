# Handwritten digit recognition — MNIST to live webcam

A digit classifier trained on MNIST, and two ways to actually point it at
handwriting: a file-based predictor and a live webcam demo that finds digits
in the frame and labels them in real time.

```bash
pip install -r requirements.txt
python train.py                      # ~99.3 % test accuracy, a few minutes on CPU
python predict_image.py numbers/*.png
python predict_webcam.py             # hold digits up to the camera, 'q' to quit
```

MNIST is downloaded and cached by Keras, so there is nothing to fetch by hand.

## The interesting problem: getting inference to match training

A model on MNIST is easy. Getting a *webcam* to work with that model is where
the real bugs are, and all of them are the same bug — the input convention
drifting between where the model was trained and where it is called.

MNIST is a specific convention: 28x28, **white digit on black**, scaled to
[0, 1]. Handwriting in the real world is the opposite — dark ink on light
paper — and every failure mode below comes from that mismatch:

- Feed a photo without inverting it, and the model sees a negative of anything
  it has ever been trained on. It does not error; it returns confident nonsense.
- Feed raw 0–255 bytes to a network trained on [0, 1] and every activation
  lands far outside its trained range.
- Crop a digit tight to its bounding box and the model sees a glyph much
  larger than MNIST's, which centres a 20x20 digit in a 28x28 frame.

The fix is structural rather than clever: the convention lives in exactly one
function, [`mnist_prep.prepare()`](mnist_prep.py), and training and both
inference paths all call it. Three copies of three lines is what let them
drift in the first place.

The webcam path has one subtlety worth calling out. It crops from the
**thresholded** image rather than the grayscale frame, because the threshold is
`THRESH_BINARY_INV` — its output is already white-on-black, i.e. already
MNIST's convention. Cropping the grayscale frame gives you the inverse, and
the demo runs perfectly while predicting garbage.

## Files

| File | |
|---|---|
| `mnist_prep.py` | The input convention, in one place. Everything else calls it. |
| `train.py` | Trains the CNN (or an MLP baseline). Reports per-digit accuracy. |
| `predict_image.py` | Classify image files. |
| `predict_webcam.py` | Live webcam detection + classification. |
| `numbers/` | A few hand-drawn test digits. |

## Models

```
python train.py --arch mlp     # 118 K params,  ~97.5 %
python train.py --arch cnn     # 1.2 M params,  ~99.3 %   (default)
python train.py --augment      # random shifts/rotations
```

`--augment` is worth trying if you are aiming at the webcam demo rather than
at the MNIST test score: MNIST digits are size-normalised and centred, real
handwriting held up to a camera is not, and small random translations and
rotations during training close some of that gap.

Training prints per-digit accuracy as well as the overall number, because a
single figure hides *which* digits get confused — 4 with 9, and 3 with 5 with
8, are the usual pairs.

## Webcam detection

Adaptive thresholding rather than a global threshold, because lighting across
a sheet of paper is rarely even. Contours are then filtered on two properties:

- **Area**, to drop specks of noise.
- **Solidity** — contour area divided by its convex hull's area. Digits are
  compact blobs with high solidity; shadows, paper edges and cable outlines
  are thin and straggly with low solidity. This is what keeps the demo from
  labelling the edge of the desk as a 7.

## What changed from the first version

The earlier version kept the Kaggle `train.csv` and `test.csv` in the
repository — 125 MB of the same data `keras.datasets.mnist` downloads for free
— along with a trained `.keras` checkpoint. Those are gone; the repo is now
about 13 KB.

Three code bugs went with them:

- `predict_image.py`'s ancestor fed un-normalised 0–255 pixels to a model
  trained on [0, 1].
- The webcam script cropped its ROI from the grayscale frame instead of the
  inverted threshold, so the digit polarity was backwards.
- `ReduceLROnPlateau` was constructed but never passed to `fit()`, and it
  monitored `val_acc`, which modern Keras does not emit — so the learning-rate
  annealer had no effect twice over.

## License

MIT — see [LICENSE](LICENSE).
