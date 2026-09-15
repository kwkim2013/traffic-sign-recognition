# Traffic Sign Recognition using CNN

CNN(Convolutional Neural Network)을 활용하여 교통표지판 이미지를 분류하는 컴퓨터 비전 프로젝트입니다.

## Project Goal

- GTSRB 교통표지판 데이터셋을 이용한 이미지 분류
- 이미지 전처리 및 데이터 증강
- TensorFlow/Keras 기반 CNN 모델 학습
- Accuracy/Loss를 이용한 학습 성능 확인
- 테스트 이미지에 대한 교통표지판 예측
- 향후 OpenCV 카메라 입력 및 자율주행 시스템과 연계

## Pipeline

`Traffic Sign Image -> Preprocessing -> CNN -> Classification -> Driving Decision`

## CNN Architecture

`32x32x3 -> Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Flatten -> Dense -> Softmax`

## Project Structure

```text
traffic-sign-recognition/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── results/
│   └── .gitkeep
├── test_images/
│   └── .gitkeep
└── src/
    ├── model.py
    ├── train.py
    └── predict.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

German Traffic Sign Recognition Benchmark (GTSRB)를 사용합니다. 데이터셋 원본은 용량이 크므로 Git 저장소에 직접 커밋하지 않고 `data/` 폴더에 로컬로 준비합니다.

학습 데이터는 클래스별 하위 폴더 구조를 사용합니다.

```text
data/train/
├── 0/
├── 1/
├── 2/
└── ...
```

## Train

```bash
python src/train.py
```

학습 완료 후 모델은 `models/traffic_sign_cnn.keras`에 저장되고 학습 그래프는 `results/training_history.png`에 저장됩니다.

## Prediction

```bash
python src/predict.py test_images/sample.png
```

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## Future Work

1. GTSRB 전체 클래스 학습 및 성능 평가
2. Confusion Matrix 추가
3. OpenCV 웹캠 실시간 입력 연동
4. 교통표지판 인식 결과를 차량 주행 명령으로 변환
5. ROS2 기반 자율주행 시스템 연계
