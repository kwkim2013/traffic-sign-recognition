"""GUI-independent TensorFlow inference and pandas result tables."""
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from labels import CLASS_NAMES


def image_tensor(image, box=None):
    if box is not None:
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError('선택 영역이 이미지 범위를 벗어났습니다.')
        image = image.crop(box)
    pixels = np.asarray(image.convert('RGB').resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
    return pixels[None, ...] / 255.0


def probability_table(probabilities):
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (43,) or not np.isfinite(values).all():
        raise ValueError('43개 클래스의 유효한 예측값이 필요합니다.')
    if (values < 0).any() or (values > 1).any() or not np.isclose(values.sum(), 1, atol=1e-3):
        raise ValueError('모델 출력은 43개 클래스의 softmax 확률이어야 합니다.')
    return pd.DataFrame({'class_id': np.arange(43), 'class_name': CLASS_NAMES,
                         'score': values}).sort_values('score', ascending=False).reset_index(drop=True)


class Predictor:
    def __init__(self):
        self.model = None
        self.model_path = None

    def run(self, model_path, image, box=None):
        stat = Path(model_path).stat()
        key = (model_path, stat.st_mtime_ns, stat.st_size)
        if self.model_path != key:
            import tensorflow as tf
            candidate = tf.keras.models.load_model(model_path, compile=False)
            if candidate.input_shape != (None, 32, 32, 3) or candidate.output_shape != (None, 43):
                raise ValueError('입력 32×32×3, 출력 43개 클래스인 GTSRB 모델을 선택하세요.')
            self.model, self.model_path = candidate, key
        output = self.model(image_tensor(image, box), training=False).numpy()[0]
        return probability_table(output)


def export_results(table, path, image_path, model_path, box):
    frame = table.copy()
    frame.insert(0, 'image_path', str(image_path))
    frame['model_path'] = str(model_path)
    frame['roi'] = str(box) if box else 'full_image'
    frame['exported_at'] = datetime.now().isoformat(timespec='seconds')
    frame.to_csv(path, index=False, encoding='utf-8-sig')
