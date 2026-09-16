"""CSV-driven loading with track-disjoint validation."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def read_rows(root, split):
    root = Path(root).resolve()
    with (root / f'{split}.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty dataset')
    for row in rows:
        path = (root / row['Path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid or missing image: {row['Path']}")
        if not 0 <= int(row['ClassId']) < 43:
            raise ValueError('ClassId must be between 0 and 42')
    return rows


def preprocess(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB').resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def load_images(root, rows):
    return np.stack([preprocess(Path(root) / r['Path']) for r in rows]), np.array([int(r['ClassId']) for r in rows], dtype=np.int32)


def track_id(row):
    parts = Path(row['Path']).stem.split('_')
    if len(parts) != 3:
        raise ValueError(f"Expected class_track_frame filename: {row['Path']}")
    return f"{row['ClassId']}_{parts[1]}"


def split_rows(rows, fraction=0.2, seed=42):
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must be in (0, 1)')
    rng = np.random.default_rng(seed)
    validation = set()
    for label in range(43):
        groups = sorted({track_id(r) for r in rows if int(r['ClassId']) == label})
        if len(groups) < 2:
            raise ValueError(f'Class {label} requires at least two tracks')
        rng.shuffle(groups)
        validation.update(groups[:max(1, min(len(groups)-1, round(len(groups)*fraction)))])
    return ([r for r in rows if track_id(r) not in validation],
            [r for r in rows if track_id(r) in validation])
