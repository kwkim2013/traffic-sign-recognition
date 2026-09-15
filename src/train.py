from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_model

IMAGE_SIZE = (32, 32)
NUM_CLASSES = 43
BATCH_SIZE = 64
EPOCHS = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"


def load_dataset(data_dir):
    images = []
    labels = []

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Training dataset was not found: {data_dir}\n"
            "Place GTSRB images in data/train/<class_id>/ before training."
        )

    for class_dir in sorted(data_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 999):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue

        class_id = int(class_dir.name)
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".ppm"}:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(class_id)

    if not images:
        raise RuntimeError("No training images were found in data/train/<class_id>/. ")

    return np.asarray(images, dtype=np.float32) / 255.0, np.asarray(labels, dtype=np.int32)


def save_training_plot(history):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("CNN Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "training_history.png", dpi=150)
    plt.close()


def main():
    print("[1/4] Loading GTSRB training images...")
    x, y = load_dataset(TRAIN_DIR)
    print(f"Loaded {len(x)} images")

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    augmentation = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    print("[2/4] Building CNN model...")
    model = build_model(input_shape=(32, 32, 3), num_classes=NUM_CLASSES)
    model.summary()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODEL_DIR / "traffic_sign_cnn.keras"

    callbacks = [
        ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ]

    print("[3/4] Training...")
    history = model.fit(
        augmentation.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    save_training_plot(history)

    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    print("[4/4] Finished")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Model: {best_model_path}")
    print(f"Graph: {RESULT_DIR / 'training_history.png'}")


if __name__ == "__main__":
    main()
