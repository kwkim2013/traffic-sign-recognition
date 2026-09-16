# GTSRB Traffic Sign Recognition — CNN

독일 교통표지판 43개 클래스를 분류하는 TensorFlow/Keras 프로젝트입니다. 기존의 3단계 CNN을 바탕으로 **데이터 준비 → 학습 → 독립 테스트 평가 → 이미지 예측** 과정을 재현할 수 있도록 보강했습니다.

## 실제 실행 성능

첨부 데이터로 학습한 모델의 **독립 테스트 정확도 93.72%, macro F1 0.9154**입니다. 12,630장 전체를 평가했습니다. 검증 정확도는 95.86%이며, 상세 설정과 한계는 [실험 기록](EXPERIMENTS.md)에 있습니다. 제공 ZIP에는 `runs/cnn/best.keras`가 포함되어 있어 설치 후 바로 예측할 수 있습니다.

## 기존 프로젝트에서 보강한 내용

| 항목 | 보강 내용 |
|---|---|
| 데이터 입력 | Kaggle Train.csv / Test.csv 기반, 중복 폴더를 제외한 ZIP 추출 |
| 검증 분할 | 클래스별 촬영 track 단위 약 80:20 분할, 연속 프레임 누수 방지 |
| 전처리 | 학습·평가·예측 모두 PIL RGB, 32×32, 0~1 정규화 |
| 데이터 증강 | 학습 중 회전·이동·확대/축소, 방향 의미가 바뀌는 좌우 반전 제외 |
| 모델 선택 | val_loss 기준 최적 체크포인트, 조기 종료, 학습률 감소 |
| 평가 | 공식 테스트셋 정확도, macro F1, 클래스별 보고서, 혼동행렬, 오분류 |
| 재현성 | seed, 실행 설정, 분할 경로, 학습 로그, 평가 모델 SHA-256 저장 |
| 예측 | 클래스 이름 및 top-k softmax 점수 출력 |

## 설치

Python 3.12 기준으로 검증합니다. 저장소 루트에서 실행하세요.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

동일한 패키지 버전으로 재현하려면 `python -m pip install -r requirements-lock.txt`를 사용하세요. 이 lock 파일은 Windows / Python 3.12.14 실행 환경에서 생성했습니다.

Linux/macOS에서는 `source .venv/bin/activate`로 활성화합니다. 활성화가 제한된 Windows 환경에서는 `.venv\Scripts\python.exe`를 직접 사용하세요. 전체 이미지가 메모리에 올라가므로 충분한 RAM이 필요합니다.

## 데이터 준비

[Kaggle GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)에서 받은 압축파일을 사용합니다.

```powershell
python src/prepare_data.py "$HOME\Downloads\archive.zip"
```

```text
data/
├── Train.csv            # 39,209개 학습 이미지의 라벨과 경로
├── Test.csv             # 12,630개 독립 테스트 이미지의 라벨과 경로
├── Meta.csv
├── Train/0/...png       # 클래스 0~42
└── Test/...png
```

첨부 ZIP에는 `Train/`과 `train/`처럼 중복된 경로가 있습니다. 준비 스크립트는 CSV가 참조하는 파일만 추출합니다. 이미지 전체 영역을 사용하며 ROI 좌표로 다시 자르지는 않습니다. 데이터와 모델은 Git에서 제외합니다. 배포 시 데이터 원본의 이용 조건을 확인하세요.

## 학습

```powershell
python src/train.py --epochs 20 --batch-size 64 --seed 42
```

기본 실행 폴더는 `runs/cnn`입니다. 기존 결과를 덮어쓰지 않으므로 재실행할 때 `--output-dir runs/experiment02`처럼 새 경로를 지정하세요. 촬영 track은 `class_track_frame.png`의 class와 track으로 식별합니다. 같은 track의 이미지가 학습/검증에 동시에 들어가지 않습니다. 클래스별 track 수를 반올림하므로 정확히 20%의 이미지가 검증에 들어가는 것은 아닙니다. Test.csv는 모델 선택에 사용하지 않습니다.

```text
RGB 32×32×3 → Augmentation (학습 시)
→ Conv 32 + MaxPool → Conv 64 + MaxPool → Conv 128 + MaxPool
→ Flatten → Dense 128 → Dropout 0.5 → Dense 43 (Softmax)
```

기존 CNN의 합성곱 블록과 분류기를 유지했습니다. Adam, sparse categorical crossentropy를 사용합니다. 최저 validation loss 모델을 저장하고 그 파일을 다시 불러와 검증합니다. 하드웨어와 라이브러리 버전이 달라지면 같은 seed에서도 결과가 다를 수 있습니다.

## 공식 테스트 평가

```powershell
python src/evaluate.py --model runs/cnn/best.keras --output-dir runs/cnn/test
```

- `metrics.json`: accuracy, macro F1, 테스트 수, 모델 해시
- `classification_report.json`: 클래스별 precision / recall / F1 / support
- `predictions.csv`: 모든 테스트 이미지의 정답·예측·점수·정답 여부
- `confusion_matrix.csv` / `.png`: 원시 건수 및 행 정규화 그림
- `misclassified.png`: 높은 점수로 틀린 사례 최대 12개 (오류가 있을 때)

클래스 불균형 때문에 정확도와 macro F1을 함께 확인합니다. 테스트 결과로 반복 튜닝하지 말고 검증셋으로 설정을 선택하세요.

## 이미지 예측

```powershell
python src/predict.py data/Test/00000.png --model runs/cnn/best.keras --top-k 3
```

출력 점수는 softmax 값이며 실제 정답 확률로 보정된 값은 아닙니다. 입력은 표지판이 잘 보이는 잘린 이미지입니다. 이 모델은 전체 도로 장면에서 표지판 위치를 찾는 검출기가 아닙니다.

## 실행 검증

```powershell
python -m unittest discover -s tests -v
python src/train.py --smoke --epochs 1 --output-dir runs/smoke
```

Smoke 실행은 클래스당 학습 8장·검증 2장으로 파이프라인 연결만 확인합니다. 이 결과를 벤치마크 성능으로 사용하지 않습니다. 실제 수행 결과는 [실험 기록](EXPERIMENTS.md)을 참고하세요.

## 파일 구성

```text
src/model.py         기존 CNN + 모델 내부 학습 전용 증강
src/data.py          공통 전처리, CSV 로딩, track 분할
src/prepare_data.py  ZIP 데이터 준비
src/train.py         학습과 최적 모델 검증
src/evaluate.py      독립 테스트 평가 및 시각화
src/predict.py       단일 이미지 top-k 예측
src/labels.py        43개 클래스의 이름
tests/               분할 누수·전처리·경로 안전성 테스트
runs/                실행별 모델·설정·결과 (Git 제외)
```

## 발표/보고서에 담을 내용

1. 문제 정의: 독일 교통표지판 43개 클래스 분류
2. 데이터 특성: 해상도·조명·시점 차이와 클래스 불균형
3. 실험 설계: track 분할, 증강, 검증셋 기반 모델 선택
4. 결과: 테스트 정확도·macro F1·혼동행렬·오분류 사례
5. 한계와 후속 실험: 저조도/흐림에 대한 강건성, 검출기 결합, 전이학습 비교

기존 코드와의 정확한 성능 비교는 동일한 데이터 분할에서 두 모델을 각각 학습해야 합니다. 이 보강만으로 성능 향상 수치를 주장하지 않습니다.

## 참고

- [기존 GitHub 프로젝트](https://github.com/kwkim2013/traffic-sign-recognition)
- [Kaggle 데이터셋](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- [TensorFlow ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)

## GitHub 반영

현재 작업본은 로컬 보강본입니다. 내용을 확인한 뒤 이 저장소 폴더에서 다음 명령으로 반영할 수 있습니다. 데이터와 runs/ 모델은 .gitignore로 제외됩니다.

```powershell
git status
git add .
git commit -m "Improve GTSRB training, evaluation and reproducibility"
git push
```

## 사진 업로드 데스크톱 UI (PySide6)

conda 환경을 활성화하고 갱신된 의존성을 설치한 뒤 실행하세요.

```powershell
conda activate traffic-sign
python -m pip install -r requirements-lock.txt
python src/gui.py
```

사진 업로드, 직접 영역 선택, TensorFlow 분류, Matplotlib Top 5 그래프, pandas CSV 저장을 지원합니다. 자세한 구조와 사용법은 [GUI 안내](GUI_GUIDE.md)를 참고하세요. 현재는 잘라낸 표지판의 종류를 분류하며 자동 위치 검출은 하지 않습니다.
