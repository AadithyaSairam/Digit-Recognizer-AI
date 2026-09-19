#!/usr/bin/env python3
"""
Train the digit classifier on MNIST.

    python train.py                      # CNN, 10 epochs
    python train.py --arch mlp --epochs 5
    python train.py --epochs 20 --augment

Loads MNIST through `keras.datasets`, so there is nothing to download by hand
and no CSV to keep in the repository, it is the same data the Kaggle
digit-recognizer CSVs contain, fetched and cached by Keras.
"""

from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from mnist_prep import IMAGE_SIZE, prepare_batch

N_CLASSES = 10


def build_mlp():
    """The simple baseline: two dense layers. Reaches ~97.5 %."""
    return keras.Sequential(
        [
            keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(N_CLASSES, activation="softmax"),
        ],
        name="mlp",
    )


def build_cnn():
    """[[Conv -> Conv -> Pool -> Dropout]] x2 -> Dense. Reaches ~99.3 %.

    The 5x5 kernels in the first block and 3x3 in the second is the standard
    shape for this problem: a wider receptive field early, where strokes are,
    and finer features later.
    """
    return keras.Sequential(
        [
            keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
            layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
            layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(N_CLASSES, activation="softmax"),
        ],
        name="cnn",
    )


ARCHITECTURES = {"mlp": build_mlp, "cnn": build_cnn}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=86)
    parser.add_argument("--out", default="digits.keras")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="small random shifts/rotations; helps on handwriting that is not "
        "centred the way MNIST is",
    )
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # The same scaling the inference scripts apply, from the same function, so
    # the two cannot drift apart.
    x_train = prepare_batch(x_train)
    x_test = prepare_batch(x_test)

    model = ARCHITECTURES[args.arch]()
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-3, rho=0.9),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # These have to be *passed to fit*. Constructing a callback and then
    # forgetting the `callbacks=` argument is silent, training runs normally
    # and the annealer simply never fires.
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            # 'val_accuracy', not 'val_acc'. The short name was removed, and
            # monitoring a metric that does not exist means the callback
            # never triggers rather than raising.
            monitor="val_accuracy",
            patience=3,
            factor=0.5,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            args.out, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    if args.augment:
        augment = keras.Sequential(
            [
                layers.RandomRotation(0.03),
                layers.RandomTranslation(0.1, 0.1),
                layers.RandomZoom(0.1),
            ],
            name="augment",
        )
        model = keras.Sequential([augment, model], name=f"{args.arch}_augmented")
        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=1e-3, rho=0.9),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=2,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss     : {loss:.4f}")
    print(f"Test accuracy : {accuracy:.4f}")

    # Per-class accuracy, because a single number hides which digits it
    # confuses, 4/9 and 3/5/8 are the usual pairs.
    predictions = model.predict(x_test, verbose=0).argmax(axis=1)
    print("\nPer-digit accuracy")
    for digit in range(N_CLASSES):
        mask = y_test == digit
        print(f"  {digit}: {100 * (predictions[mask] == digit).mean():.2f} %  (n={mask.sum()})")

    print(f"\nSaved best checkpoint to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
