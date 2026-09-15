# Dataset

이 폴더에는 GTSRB (German Traffic Sign Recognition Benchmark) 학습 데이터를 배치합니다.

프로젝트의 기본 학습 코드는 다음 구조를 기대합니다.

```text
data/
└── train/
    ├── 0/
    │   ├── image1.png
    │   └── ...
    ├── 1/
    ├── 2/
    └── ...
```

각 폴더 이름은 GTSRB class ID이며 전체 분류 클래스 수는 43개입니다.

데이터셋 이미지 파일은 용량 때문에 GitHub 저장소에 커밋하지 않습니다.
