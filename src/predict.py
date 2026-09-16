"""Predict class names and top-k scores for one cropped traffic sign."""
import argparse
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from data import ROOT, preprocess
from labels import CLASS_NAMES

DEFAULT_MODEL = ROOT / 'runs/cnn/best.keras'


def preprocess_image(image_path):
    return np.expand_dims(preprocess(image_path), axis=0)


def predict(image_path, model_path):
    probabilities = load_model(model_path)(preprocess_image(image_path), training=False).numpy()[0]
    class_id = int(np.argmax(probabilities))
    return class_id, float(probabilities[class_id])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--top-k', type=int, default=3, choices=range(1, 44))
    args = p.parse_args()
    probabilities = load_model(args.model)(preprocess_image(args.image), training=False).numpy()[0]
    for i in np.argsort(probabilities)[::-1][:args.top_k]:
        print(f'{i:2d} | {CLASS_NAMES[i]} | {probabilities[i]:.2%}')


if __name__ == '__main__':
    main()
