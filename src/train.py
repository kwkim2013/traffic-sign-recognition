"""Train the original CNN with reproducible track-disjoint validation."""
import argparse
import json
import platform
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from data import ROOT, DATA, read_rows, split_rows, load_images
from model import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=DATA)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'runs' / 'cnn')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke', action='store_true', help='Small pipeline check; not a benchmark')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error('epochs and batch-size must be positive')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    train, val = split_rows(read_rows(args.data_dir, 'Train'), seed=args.seed)
    if args.smoke:
        train = [r for c in range(43) for r in [r for r in train if int(r['ClassId']) == c][:8]]
        val = [r for c in range(43) for r in [r for r in val if int(r['ClassId']) == c][:2]]
    (out / 'split.json').write_text(json.dumps({'train': [r['Path'] for r in train], 'validation': [r['Path'] for r in val]}, indent=2), encoding='utf-8')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(python=platform.python_version(), tensorflow=tf.__version__, train_count=len(train), validation_count=len(val))
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    x, y = load_images(args.data_dir, train)
    xv, yv = load_images(args.data_dir, val)
    model = build_model()
    callbacks = [tf.keras.callbacks.ModelCheckpoint(str(out / 'best.keras'), monitor='val_loss', save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5),
                 tf.keras.callbacks.CSVLogger(str(out / 'history.csv'))]
    history = model.fit(x, y, validation_data=(xv, yv), batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks, verbose=2)
    best = tf.keras.models.load_model(out / 'best.keras')
    metrics = best.evaluate(xv, yv, verbose=0, return_dict=True)
    (out / 'validation.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ['accuracy', 'loss']):
        ax.plot(history.history[metric], label='Train')
        ax.plot(history.history['val_' + metric], label='Validation')
        ax.set(xlabel='Epoch (zero-based)', ylabel=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'history.png', dpi=150)
    plt.close(fig)
    print(json.dumps(metrics), out)


if __name__ == '__main__':
    main()
