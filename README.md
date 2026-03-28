# 🎤 AI Singer — KK's Voice Clone Pipeline

> RVC(Retrieval-based Voice Conversion) 기반 AI 보컬 생성 파이프라인.  
> 나의 목소리를 학습시켜, AI가 노래하게 만드는 프로젝트.

---

## 🎯 프로젝트 소개

이 프로젝트는 개인 보컬 샘플을 기반으로 AI 가수를 생성하는 엔드투엔드 파이프라인입니다.

- 🎙️ **보컬 학습** — 소량의 음성 샘플(WAV)로 개인 음색 모델 학습
- 🔄 **음성 변환** — 학습된 모델로 임의의 오디오를 내 목소리로 변환
- 🎵 **데모곡 생성** — AI 보컬 + 악기 트랙 합성으로 완성곡 생성

---

## ⚡ 빠른 시작

```bash
# 1. 클론
git clone https://github.com/songkang71/ai-singer.git
cd ai-singer

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 보컬 샘플 준비 (dataset/raw/ 에 WAV 파일 배치)
mkdir -p dataset/raw
# cp your_vocal_samples.wav dataset/raw/

# 4. 모델 학습
python scripts/train.py

# 5. 음성 변환
python scripts/convert.py

# 6. 데모곡 생성
python scripts/demo.py
```

---

## 🔧 파이프라인 흐름

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  보컬 샘플   │───▶│  모델 학습   │───▶│  학습된 모델  │
│  (WAV 파일)  │    │  (RVC 기반)  │    │  (.pth)      │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                    ┌──────────────┐            │
                    │  원본 오디오  │────────────┤
                    │  (멜로디)    │            │
                    └──────────────┘            ▼
                                        ┌──────────────┐
                                        │  음성 변환   │
                                        │  (Voice      │
                                        │   Conversion)│
                                        └──────┬───────┘
                                               │
                    ┌──────────────┐            │
                    │  악기 트랙   │────────────┤
                    │  (MR)       │            │
                    └──────────────┘            ▼
                                        ┌──────────────┐
                                        │  최종 믹싱   │
                                        │  → 완성곡 🎵  │
                                        └──────────────┘
```

---

## 📂 프로젝트 구조

```
ai-singer/
├── README.md               # 이 파일
├── LICENSE                  # MIT License
├── requirements.txt         # Python 의존성
├── configs/
│   ├── kk_vocal_config.json # 보컬 모델 설정
│   └── fast_training.yaml   # 학습 파라미터 설정
├── scripts/
│   ├── train.py             # 🔥 고품질 모델 학습 (100 에포크)
│   ├── train_basic.py       # 기본 모델 학습 (30 에포크)
│   ├── convert.py           # 🎤 음성 변환 (학습된 모델 → 보컬 생성)
│   ├── generate_full.py     # 🎵 풀 버전 데모곡 생성
│   ├── generate_simple.py   # 간편 버전 데모곡 생성
│   └── demo.py              # ⚡ 빠른 데모 생성
└── examples/
    └── README.md            # 출력 예시 설명
```

---

## 🛠️ 기술 스택

| 기술 | 용도 |
|------|------|
| **PyTorch** | 딥러닝 모델 학습 및 추론 |
| **RVC** | Retrieval-based Voice Conversion 프레임워크 |
| **librosa** | 오디오 분석 (스펙트로그램, 멜 밴드, FFT) |
| **soundfile** | WAV 파일 읽기/쓰기 |
| **Apple MPS** | Mac M-시리즈 GPU 가속 (자동 감지) |

---

## 📊 학습 결과

| 메트릭 | 기본 학습 | 고품질 학습 |
|--------|----------|------------|
| **에포크** | 30 | 100 |
| **손실 개선** | 87% | 98.6% |
| **오디오 품질** | 48kHz | 96kHz |
| **멜 밴드** | 80 | 128 |
| **소요 시간** | ~2초 | ~15초 |

---

## 💡 사용 팁

1. **보컬 샘플은 깨끗할수록 좋습니다** — 반주 없이 목소리만 녹음
2. **최소 3개 이상의 샘플** 권장 (다양한 음역대)
3. **고품질 학습(`train.py`)** 을 기본으로 사용하세요
4. **Mac M-시리즈**에서 MPS 가속이 자동 활성화됩니다

---

## 📄 License

MIT License — 자유롭게 사용, 수정, 배포할 수 있습니다.

---

## 🙏 Credits

- [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — 핵심 음성 변환 프레임워크
- PyTorch — 딥러닝 프레임워크
- librosa — 오디오 분석 라이브러리

---

*Built with 🔥 by KK & Chloe 🦞*
