"""Evaluate a saved checkpoint against the official Test.csv labels."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from data import DATA, ROOT, read_rows, load_images
from labels import CLASS_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=DATA)
    p.add_argument('--model', type=Path, default=ROOT / 'runs/cnn/best.keras')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'runs/cnn/test')
    p.add_argument('--batch-size', type=int, default=128)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch-size must be positive')
    rows = read_rows(args.data_dir, 'Test')
    model = tf.keras.models.load_model(args.model)
    probs = []
    for start in range(0, len(rows), args.batch_size):
        x, _ = load_images(args.data_dir, rows[start:start+args.batch_size])
        probs.append(model(x, training=False).numpy())
    probs = np.concatenate(probs)
    y = np.array([int(r['ClassId']) for r in rows])
    pred = probs.argmax(axis=1)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    labels = list(range(43))
    metrics = dict(accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, labels=labels, average='macro', zero_division=0),
                   test_count=len(y), model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest())
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    report = classification_report(y, pred, labels=labels, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    (out / 'classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'true_class', 'predicted_class', 'class_name', 'confidence', 'correct'])
        for row, actual, predicted, probability in zip(rows, y, pred, probs):
            writer.writerow([row['Path'], actual, predicted, CLASS_NAMES[predicted], float(probability[predicted]), bool(actual == predicted)])
    cm = confusion_matrix(y, pred, labels=labels)
    np.savetxt(out / 'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), vmin=0, vmax=1, cmap='Blues')
    ax.set(xticks=labels, yticks=labels, xlabel='Predicted class ID', ylabel='True class ID', title='GTSRB: row-normalized confusion matrix')
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    wrong = np.flatnonzero(y != pred)
    selected = sorted(wrong, key=lambda i: -float(probs[i, pred[i]]))[:12]
    if selected:
        from PIL import Image
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for ax in axes.flat:
            ax.axis('off')
        for ax, i in zip(axes.flat, selected):
            with Image.open(args.data_dir / rows[i]['Path']) as image:
                ax.imshow(image.convert('RGB'))
            ax.set_title(f'True: {y[i]} / Pred: {pred[i]}\nScore: {probs[i, pred[i]]:.3f}')
        fig.tight_layout()
        fig.savefig(out / 'misclassified.png', dpi=150)
        plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
