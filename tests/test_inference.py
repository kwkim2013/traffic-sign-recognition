import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from inference import image_tensor, probability_table, export_results


class InferenceTests(unittest.TestCase):
    def test_roi_and_normalization(self):
        image = Image.new('RGB', (40, 20), 'red')
        image.paste('blue', (20, 0, 40, 20))
        tensor = image_tensor(image, (20, 0, 40, 20))
        self.assertEqual(tensor.shape, (1, 32, 32, 3))
        np.testing.assert_array_equal(tensor[0, 0, 0], [0, 0, 1])
        with self.assertRaises(ValueError):
            image_tensor(image, (10, 10, 5, 5))

    def test_order_and_export(self):
        scores = np.zeros(43)
        scores[14] = .8
        scores[13] = .2
        frame = probability_table(scores)
        self.assertEqual(frame.iloc[0].class_name, 'Stop')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.csv'
            export_results(frame, path, '표지판.png', 'best.keras', None)
            saved = pd.read_csv(path)
            self.assertEqual(len(saved), 43)
            self.assertEqual(saved.iloc[0].image_path, '표지판.png')
            self.assertEqual(saved.iloc[0].class_id, 14)

    def test_reject_invalid_outputs(self):
        for values in (np.zeros(42), np.zeros(43), np.full(43, np.nan)):
            with self.assertRaises(ValueError):
                probability_table(values)
