"""Extract CSV-referenced images without duplicate lowercase folders."""
import argparse
import csv
import io
from pathlib import Path
import shutil
import zipfile


def extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        names = ['Train.csv', 'Test.csv', 'Meta.csv']
        for split in ('Train', 'Test'):
            rows = csv.DictReader(io.StringIO(z.read(f'{split}.csv').decode('utf-8-sig')))
            names.extend(r['Path'] for r in rows)
        members = set(z.namelist())
        for name in names:
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or name not in members:
                raise ValueError(f'Unsafe or missing member: {name}')
        for name in names:
            target = destination / name
            if target.exists():
                if target.stat().st_size != z.getinfo(name).file_size:
                    raise FileExistsError(f'Existing file differs: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
    print(f'Prepared {len(names)} files in {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    args = parser.parse_args()
    extract(args.archive, args.data_dir)
