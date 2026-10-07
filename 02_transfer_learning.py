"""
CS5720 Neural Network and Deep Learning - Home Assignment 3
Q2: Transfer learning, freeze vs fine-tune

Experiment A freezes ResNet50 and trains a new classifier.
Experiment B unfreezes the last convolutional block (conv5) and trains
that block with the classifier. Both runs use the same data and epochs.
"""

import json
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CIFAR-10 is 32x32. 96x96 gives the pretrained stride-32 network a 3x3
# map before global pooling, and it still fits a laptop CPU.
IMG_SIZE = 96
# Airplane and automobile. Two classes meet the assignment, and the subset
# keeps the two ResNet50 runs short enough to finish on CPU.
CLASS_IDS = (0, 1)
CLASS_NAMES = ("airplane", "automobile")
TRAIN_PER_CLASS = 1500
VAL_PER_CLASS = 300
TEST_PER_CLASS = 400
EPOCHS = 5
BATCH_SIZE = 32
# The new head can take a larger step. The pretrained block needs a smaller one.
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4
LAST_BLOCK_PREFIX = "conv5_"


def parameter_count(weights):
    return int(sum(np.prod(weight.shape) for weight in weights))


def sample_class(images, labels, class_id, count, rng, excluded=None):
    """Sample `count` images of one class. `excluded` indexes are left out."""
    index = np.flatnonzero(labels == class_id)
    if excluded is not None:
        index = np.setdiff1d(index, excluded, assume_unique=True)
    if count > index.size:
        raise ValueError(f"class {class_id} has {index.size} images left, need {count}")
    chosen = rng.choice(index, size=count, replace=False)
    return images[chosen], chosen


def stack_and_shuffle(parts, labels, rng):
    images = np.concatenate(parts, axis=0)
    label_array = np.concatenate(labels, axis=0)
    order = rng.permutation(label_array.shape[0])
    return images[order], label_array[order]


def resize_and_preprocess(images):
    """Resize to the ResNet input and apply the ImageNet BGR mean subtraction."""
    resized = tf.image.resize(images.astype("float32"), (IMG_SIZE, IMG_SIZE))
    return preprocess_input(resized.numpy())


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    rng = np.random.default_rng(42)

    # Train and validation are disjoint slices of the CIFAR-10 training set.
    # The test images come from the official test split.
    train_parts = []
    val_parts = []
    train_labels = []
    val_labels = []
    test_parts = []
    test_labels = []
    for class_id in CLASS_IDS:
        train_images, train_index = sample_class(
            x_train, y_train, class_id, TRAIN_PER_CLASS, rng
        )
        val_images, _ = sample_class(
            x_train, y_train, class_id, VAL_PER_CLASS, rng, excluded=train_index
        )
        test_images, _ = sample_class(x_test, y_test, class_id, TEST_PER_CLASS, rng)
        train_parts.append(train_images)
        val_parts.append(val_images)
        test_parts.append(test_images)
        train_labels.append(np.full(TRAIN_PER_CLASS, class_id, dtype=np.int32))
        val_labels.append(np.full(VAL_PER_CLASS, class_id, dtype=np.int32))
        test_labels.append(np.full(TEST_PER_CLASS, class_id, dtype=np.int32))

    train_images, train_labels = stack_and_shuffle(train_parts, train_labels, rng)
    val_images, val_labels = stack_and_shuffle(val_parts, val_labels, rng)
    test_images, test_labels = stack_and_shuffle(test_parts, test_labels, rng)

    return {
        "x_train": resize_and_preprocess(train_images),
        "y_train": train_labels,
        "x_val": resize_and_preprocess(val_images),
        "y_val": val_labels,
        "x_test": resize_and_preprocess(test_images),
        "y_test": test_labels,
    }


def set_trainable(base, mode):
    """Freeze the base, or open only the conv5 block.

    Batch-norm stays frozen in both modes. With a small batch, updating the
    ImageNet running statistics hurts more than it helps. The conv kernels
    in conv5 still train in the fine-tune run.
    """
    if mode == "frozen":
        base.trainable = False
        return

    if mode != "finetune":
        raise ValueError(f"unknown mode: {mode}")

    base.trainable = True
    opened = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        elif layer.name.startswith(LAST_BLOCK_PREFIX):
            layer.trainable = True
            opened = True
        else:
            layer.trainable = False
    if not opened:
        raise RuntimeError("ResNet50 has no conv5_ layers to unfreeze")


def build_model(mode, learning_rate):
    # New head, same pretrained trunk, for each experiment.
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    base = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    set_trainable(base, mode)

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # Frozen run: keep every base layer, including batch-norm, in inference.
    if mode == "frozen":
        features = base(inputs, training=False)
    else:
        features = base(inputs)
    outputs = tf.keras.layers.Dense(
        len(CLASS_IDS),
        activation="softmax",
        name="classifier",
    )(features)
    model = tf.keras.Model(inputs, outputs, name=f"resnet50_{mode}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def trainable_layer_lines(model):
    """Name each trainable layer that owns weights, including layers inside ResNet50."""
    lines = []
    for layer in model.layers:
        children = getattr(layer, "layers", None)
        if children:
            for child in children:
                count = parameter_count(child.trainable_weights) if child.trainable else 0
                if count:
                    lines.append(f"{child.name}: {count:,}")
        else:
            count = parameter_count(layer.trainable_weights) if layer.trainable else 0
            if count:
                lines.append(f"{layer.name}: {count:,}")
    return lines


def run_experiment(mode, learning_rate, data):
    model = build_model(mode, learning_rate)
    trainable = parameter_count(model.trainable_weights)
    total = parameter_count(model.weights)
    print(f"\n{mode}: trainable parameters = {trainable:,} / {total:,}")

    start = time.perf_counter()
    history = model.fit(
        data["x_train"],
        data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=2,
    )
    train_seconds = time.perf_counter() - start

    test_loss, test_accuracy = model.evaluate(
        data["x_test"],
        data["y_test"],
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    print(
        f"{mode}: test accuracy = {test_accuracy:.4f}, "
        f"test loss = {test_loss:.4f}, time = {train_seconds:.1f}s"
    )

    history_dict = {
        key: [float(value) for value in values] for key, values in history.history.items()
    }
    return {
        "mode": mode,
        "learning_rate": learning_rate,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_seconds": float(train_seconds),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs": EPOCHS,
        "history": history_dict,
        "trainable_weights": trainable_layer_lines(model),
    }


def plot_training_loss(frozen, finetune, path):
    epochs = range(1, EPOCHS + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, frozen["history"]["loss"], marker="o", label="Frozen feature extractor")
    plt.plot(epochs, finetune["history"]["loss"], marker="o", label="Fine-tuned network")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training loss: freeze vs fine-tune")
    plt.xticks(list(epochs))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_report(frozen, finetune, plot_path):
    def row(result):
        return (
            f"{result['trainable_parameters']:,}".ljust(24)
            + f"{result['train_seconds']:.1f} s".ljust(16)
            + f"{result['test_accuracy']:.4f}"
        )

    lines = [
        "Q2: Freeze vs fine-tune",
        "",
        "Model: ResNet50, ImageNet weights, include_top=False, global average pooling",
        "New classifier: Dense(2), softmax",
        f"Dataset: CIFAR-10 classes {CLASS_NAMES[0]} ({CLASS_IDS[0]}) and {CLASS_NAMES[1]} ({CLASS_IDS[1]})",
        f"Input: resized from 32x32 to {IMG_SIZE}x{IMG_SIZE}, ResNet50 preprocess_input",
        (
            f"Train / val / test per class: {TRAIN_PER_CLASS} / {VAL_PER_CLASS} / {TEST_PER_CLASS}"
        ),
        f"Epochs: {EPOCHS}, batch size: {BATCH_SIZE}",
        f"Frozen learning rate: {HEAD_LEARNING_RATE}",
        f"Fine-tune learning rate: {FINE_TUNE_LEARNING_RATE}",
        "Fine-tune trains conv5 kernels. Batch-norm stays frozen in both runs.",
        "Training time is the wall-clock time of model.fit. Accuracy is test accuracy.",
        "",
        "Method                    Trainable Parameters  Training Time   Test Accuracy",
        "Frozen Feature Extractor  " + row(frozen),
        "Fine-Tuned Network        " + row(finetune),
        "",
        "Frozen training loss: "
        + ", ".join(f"{value:.4f}" for value in frozen["history"]["loss"]),
        "Fine-tune training loss: "
        + ", ".join(f"{value:.4f}" for value in finetune["history"]["loss"]),
        "Frozen validation accuracy: "
        + ", ".join(f"{value:.4f}" for value in frozen["history"]["val_accuracy"]),
        "Fine-tune validation accuracy: "
        + ", ".join(f"{value:.4f}" for value in finetune["history"]["val_accuracy"]),
        "",
        f"Plot: {plot_path}",
        "",
        "Frozen trainable weights:",
    ]
    lines.extend(f"  {name}" for name in frozen["trainable_weights"])
    lines.append("")
    lines.append("Fine-tune trainable weights:")
    lines.extend(f"  {name}" for name in finetune["trainable_weights"])
    lines.append("")

    text = "\n".join(lines)
    report_path = os.path.join(OUTPUT_DIR, "q2_comparison.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("\n" + text)

    metrics = {
        "image_size": IMG_SIZE,
        "class_names": list(CLASS_NAMES),
        "train_per_class": TRAIN_PER_CLASS,
        "val_per_class": VAL_PER_CLASS,
        "test_per_class": TEST_PER_CLASS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "plot": os.path.basename(plot_path),
        "frozen": {key: value for key, value in frozen.items() if key != "trainable_weights"},
        "finetune": {
            key: value for key, value in finetune.items() if key != "trainable_weights"
        },
    }
    metrics_path = os.path.join(OUTPUT_DIR, "q2_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved {report_path}")
    print(f"Saved {metrics_path}")


def main():
    print("=" * 60)
    print("Q2: Transfer learning, freeze vs fine-tune")
    print("=" * 60)
    data = load_data()
    print(
        "Splits:",
        data["x_train"].shape[0],
        "train,",
        data["x_val"].shape[0],
        "val,",
        data["x_test"].shape[0],
        "test",
    )

    frozen = run_experiment("frozen", HEAD_LEARNING_RATE, data)
    finetune = run_experiment("finetune", FINE_TUNE_LEARNING_RATE, data)

    plot_path = os.path.join(OUTPUT_DIR, "q2_training_loss.png")
    plot_training_loss(frozen, finetune, plot_path)
    print(f"Saved {plot_path}")
    write_report(frozen, finetune, plot_path)


if __name__ == "__main__":
    main()
