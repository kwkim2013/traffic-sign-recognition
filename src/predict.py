import argparse
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "traffic_sign_cnn.keras"
IMAGE_SIZE = (32, 32)


def preprocess_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)


def predict(image_path, model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model was not found: {model_path}\nRun python src/train.py first."
        )

    model = load_model(model_path)
    image = preprocess_image(image_path)

    probabilities = model.predict(image, verbose=0)[0]
    class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[class_id])

    return class_id, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict a GTSRB traffic sign using a trained CNN.")
    parser.add_argument("image", type=Path, help="Path to a traffic-sign image")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to .keras model")
    args = parser.parse_args()

    class_id, confidence = predict(args.image, args.model)

    print(f"Predicted class ID: {class_id}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
