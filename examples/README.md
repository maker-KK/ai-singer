# 📦 Examples & Demo Outputs

이 디렉토리는 파이프라인 실행 후 생성되는 출력물의 예시입니다.

## 실행하면 생기는 것들

### 학습 후 (`scripts/train.py`)
```
models/
└── kk_high_quality_vocal_model.pth    # 학습된 보컬 모델 (~13MB)
```

### 변환 후 (`scripts/convert.py`)
```
converted_results/
├── converted_sample_1.wav             # 변환된 보컬
├── converted_sample_2.wav
└── conversion_results.json            # 변환 메타데이터
```

### 데모 생성 후 (`scripts/demo.py`)
```
final_ai_vocal_demo/
├── kk_ai_vocal_demo.wav              # AI 보컬 데모 (5초)
└── demo_metadata.json                # 데모 정보
```

### 풀 버전 생성 후 (`scripts/generate_full.py`)
```
full_version_digital_nomad/
├── ai_vocals/                         # 섹션별 AI 보컬
│   ├── verse_vocal.wav
│   ├── chorus_vocal.wav
│   └── bridge_vocal.wav
├── instrumentals/                     # 악기 트랙
│   ├── guitar.wav
│   ├── piano.wav
│   └── drums.wav
└── final_song/
    └── digital_nomad_full.wav         # 최종 완성곡 🎵
```

## 참고

- `.wav`, `.pth` 등 바이너리 파일은 `.gitignore`에 의해 Git에 포함되지 않습니다.
- 직접 파이프라인을 실행하여 결과물을 생성하세요.
