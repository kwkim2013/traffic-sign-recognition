import sys
import unittest
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import split_rows, track_id, preprocess
from prepare_data import extract


class PipelineTests(unittest.TestCase):
    def test_tracks_never_cross_and_all_classes_present(self):
        rows = [dict(ClassId=str(c), Path=f'Train/{c}/{c:05}_{t:05}_{f:05}.png')
                for c in range(43) for t in range(5) for f in range(3)]
        train, val = split_rows(rows)
        self.assertFalse({track_id(r) for r in train} & {track_id(r) for r in val})
        self.assertEqual(len(train) + len(val), len(rows))
        self.assertEqual({r['ClassId'] for r in val}, {str(c) for c in range(43)})
        self.assertEqual((train, val), split_rows(rows))

    def test_rgb_normalization(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'red.png'
            Image.new('RGB', (9, 17), (255, 0, 0)).save(path)
            x = preprocess(path)
            self.assertEqual(x.shape, (32, 32, 3))
            np.testing.assert_array_equal(x[0, 0], [1, 0, 0])

    def test_reject_zip_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.zip'
            with zipfile.ZipFile(path, 'w') as z:
                for split in ('Train', 'Test'):
                    z.writestr(f'{split}.csv', 'ClassId,Path\n0,../escape.png\n')
                z.writestr('Meta.csv', '')
                z.writestr('../escape.png', 'bad')
            with self.assertRaises(ValueError):
                extract(path, Path(d) / 'data')
            self.assertFalse((Path(d) / 'escape.png').exists())


if __name__ == '__main__':
    unittest.main()
