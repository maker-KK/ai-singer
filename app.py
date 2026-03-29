"""
🎤 AI Singer Studio — Gradio GUI
보컬 업로드 → 분석 → 학습 → 변환 → 재생까지 한 화면에서
"""
import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import os
import json
import time
import tempfile
import urllib.request
import urllib.error
import urllib.parse
import http.client
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# ─── 모델 정의 ───────────────────────────────────
class RVCModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, input_dim, 5, padding=2),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─── 디렉토리 ────────────────────────────────────
BASE = Path(__file__).parent
DATASET_DIR = BASE / "dataset" / "raw"
MODEL_DIR = BASE / "models"
OUTPUT_DIR = BASE / "outputs"
SUNO_OUTPUT_DIR = BASE / "outputs" / "suno"
for d in [DATASET_DIR, MODEL_DIR, OUTPUT_DIR, SUNO_OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Suno 설정 ───────────────────────────────────
# Suno 곡은 suno.com에서 직접 생성 후 URL로 가져옴

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ─── Tab 1: 보컬 업로드 & 분석 ───────────────────
def analyze_vocal(audio_file):
    if audio_file is None:
        return None, "⚠️ 파일을 업로드하세요.", None

    # 파일 저장
    src = Path(audio_file)
    dst = DATASET_DIR / src.name
    import shutil
    shutil.copy2(src, dst)

    audio, sr = sf.read(str(dst))
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    duration = len(audio) / sr

    # 분석
    pitches, mag = librosa.piptrack(y=audio.astype(np.float32), sr=sr)
    pitch_vals = []
    for t in range(pitches.shape[1]):
        idx = mag[:, t].argmax()
        p = pitches[idx, t]
        if p > 50:
            pitch_vals.append(p)

    avg_pitch = np.mean(pitch_vals) if pitch_vals else 0
    min_pitch = np.min(pitch_vals) if pitch_vals else 0
    max_pitch = np.max(pitch_vals) if pitch_vals else 0

    # 스펙트로그램 이미지
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # 파형
    times = np.arange(len(audio)) / sr
    axes[0].plot(times, audio, linewidth=0.3, color="#4A90D9")
    axes[0].set_title("Waveform", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    # 멜 스펙트로그램
    mel = librosa.feature.melspectrogram(y=audio.astype(np.float32), sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1], cmap="magma")
    axes[1].set_title("Mel Spectrogram", fontsize=12, fontweight="bold")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    fig_path = tempfile.mktemp(suffix=".png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    info = f"""### 📊 보컬 분석 결과

| 항목 | 값 |
|------|-----|
| **파일** | `{src.name}` |
| **길이** | {duration:.1f}초 |
| **샘플레이트** | {sr:,} Hz |
| **평균 피치** | {avg_pitch:.0f} Hz |
| **음역대** | {min_pitch:.0f} ~ {max_pitch:.0f} Hz |
| **저장 위치** | `{dst}` |

✅ 학습 데이터로 등록되었습니다. **모델 학습** 탭에서 학습을 시작하세요.
"""
    # 업로드된 파일 목록
    files = list(DATASET_DIR.glob("*.wav"))
    file_list = "\n".join([f"- `{f.name}` ({f.stat().st_size / 1024:.0f}KB)" for f in files])
    info += f"\n\n### 📁 등록된 보컬 샘플 ({len(files)}개)\n{file_list}"

    return fig_path, info, str(dst)


# ─── Tab 2: 모델 학습 ────────────────────────────
def train_model(epochs, learning_rate, quality, progress=gr.Progress()):
    files = list(DATASET_DIR.glob("*.wav"))
    if not files:
        return None, "❌ 학습할 보컬 샘플이 없습니다. 먼저 보컬을 업로드하세요."

    n_mels = 128 if quality == "고품질 (128 mel)" else 80
    model = RVCModel(input_dim=n_mels, hidden_dim=512 if quality == "고품질 (128 mel)" else 256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 특징 추출
    progress(0, desc="🎤 보컬 특징 추출 중...")
    features = []
    for f in files:
        audio, sr = sf.read(str(f))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        mel = librosa.feature.melspectrogram(
            y=audio.astype(np.float32), sr=sr, n_mels=n_mels, n_fft=4096, hop_length=256
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        seq_len = 200
        for i in range(mel_db.shape[1] // seq_len):
            seg = mel_db[:, i * seq_len:(i + 1) * seq_len]
            if seg.shape[1] == seq_len:
                features.append(seg)

    if not features:
        return None, "❌ 특징 추출 실패. 보컬 샘플이 너무 짧을 수 있습니다."

    # 학습 루프
    losses = []
    best_loss = float("inf")
    epochs = int(epochs)
    start = time.time()

    for epoch in range(epochs):
        np.random.shuffle(features)
        epoch_loss = 0
        n_batches = 0
        batch_size = 4

        for i in range(0, len(features) - batch_size + 1, batch_size):
            batch = [torch.FloatTensor(features[j]).unsqueeze(0) for j in range(i, i + batch_size)]
            source = torch.cat(batch, dim=0).to(DEVICE)
            target = source + torch.randn_like(source) * 0.02

            optimizer.zero_grad()
            output = model(source)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), str(MODEL_DIR / "best_model.pth"))

        pct = (epoch + 1) / epochs
        progress(pct, desc=f"에포크 {epoch+1}/{epochs} — 손실: {avg:.6f}")

    elapsed = time.time() - start

    # 최종 모델 저장
    final_path = MODEL_DIR / "kk_vocal_model.pth"
    torch.save(model.state_dict(), str(final_path))

    # 손실 그래프
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(losses) + 1), losses, color="#E74C3C", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training Loss", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig_path = tempfile.mktemp(suffix=".png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    improvement = (losses[0] - best_loss) / losses[0] * 100 if losses[0] > 0 else 0
    params = sum(p.numel() for p in model.parameters())

    info = f"""### ✅ 학습 완료

| 항목 | 값 |
|------|-----|
| **에포크** | {epochs} |
| **초기 손실** | {losses[0]:.6f} |
| **최종 손실** | {losses[-1]:.6f} |
| **최고 손실** | {best_loss:.6f} |
| **개선률** | {improvement:.1f}% |
| **소요 시간** | {elapsed:.1f}초 |
| **모델 파라미터** | {params:,}개 |
| **디바이스** | {DEVICE} |
| **저장 위치** | `{final_path}` |
"""
    return fig_path, info


# ─── Tab 3: 음성 변환 ────────────────────────────
def convert_voice(audio_file, pitch_shift):
    if audio_file is None:
        return None, "⚠️ 변환할 오디오 파일을 업로드하세요."

    model_path = MODEL_DIR / "kk_vocal_model.pth"
    if not model_path.exists():
        model_path = MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        return None, "❌ 학습된 모델이 없습니다. 먼저 모델 학습을 진행하세요."

    # 오디오 로드
    audio, sr = sf.read(audio_file)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # 모델 로드 + 추론
    model = RVCModel().to(DEVICE)
    model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
    model.eval()

    # 멜 스펙트로그램 → 모델 → 역변환
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=sr, n_mels=128, n_fft=4096, hop_length=256
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    with torch.no_grad():
        inp = torch.FloatTensor(mel_db).unsqueeze(0).to(DEVICE)
        out = model(inp).cpu().numpy()[0]

    # Griffin-Lim 복원
    mel_linear = librosa.db_to_power(out)
    inv_mel = librosa.filters.mel(sr=sr, n_fft=4096, n_mels=128)
    spec = np.dot(np.linalg.pinv(inv_mel), mel_linear)
    converted = librosa.griffinlim(spec, n_iter=64, hop_length=256, n_fft=4096)

    # 피치 시프트
    if pitch_shift != 0:
        converted = librosa.effects.pitch_shift(converted, sr=sr, n_steps=pitch_shift)

    # 정규화
    converted = converted / (np.max(np.abs(converted)) + 1e-8) * 0.9

    # 저장
    out_path = OUTPUT_DIR / f"converted_{datetime.now().strftime('%H%M%S')}.wav"
    sf.write(str(out_path), converted, sr)

    info = f"""### 🎤 변환 완료

| 항목 | 값 |
|------|-----|
| **입력 길이** | {len(audio)/sr:.1f}초 |
| **출력 길이** | {len(converted)/sr:.1f}초 |
| **피치 시프트** | {pitch_shift:+d} 반음 |
| **샘플레이트** | {sr:,} Hz |
| **저장 위치** | `{out_path}` |
"""
    return str(out_path), info


# ─── Tab 4: 데모곡 생성 ──────────────────────────
def generate_demo(style, duration_sec, progress=gr.Progress()):
    model_path = MODEL_DIR / "kk_vocal_model.pth"
    if not model_path.exists():
        model_path = MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        return None, None, "❌ 학습된 모델이 없습니다."

    progress(0.1, desc="🎹 악기 트랙 생성 중...")
    sr = 48000
    t = np.linspace(0, duration_sec, int(sr * duration_sec))

    # 스타일별 파라미터
    styles = {
        "발라드": {"bpm": 72, "freqs": [261.63, 329.63, 392.00, 523.25], "pad_freq": 130.81},
        "팝": {"bpm": 120, "freqs": [329.63, 392.00, 440.00, 523.25], "pad_freq": 164.81},
        "일렉트로닉": {"bpm": 128, "freqs": [440.00, 523.25, 659.26, 783.99], "pad_freq": 220.00},
        "락": {"bpm": 140, "freqs": [196.00, 246.94, 329.63, 392.00], "pad_freq": 98.00},
    }
    params = styles.get(style, styles["팝"])

    # 멜로디
    melody = np.zeros_like(t)
    beat_dur = 60.0 / params["bpm"]
    for i, freq in enumerate(params["freqs"] * (int(duration_sec / (beat_dur * len(params["freqs"]))) + 1)):
        start = int(i * beat_dur * sr)
        end = min(start + int(beat_dur * sr), len(t))
        if start < len(t):
            seg_t = np.arange(end - start) / sr
            envelope = np.exp(-seg_t * 2.0)
            melody[start:end] += np.sin(2 * np.pi * freq * seg_t) * envelope * 0.3

    # 패드
    pad = np.sin(2 * np.pi * params["pad_freq"] * t) * 0.1

    progress(0.4, desc="🎤 AI 보컬 생성 중...")

    # 모델로 보컬 생성
    model = RVCModel().to(DEVICE)
    model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
    model.eval()

    vocal_len = min(len(t), sr * duration_sec)
    noise = np.random.randn(128, int(vocal_len / 256) + 1).astype(np.float32) * 0.5

    with torch.no_grad():
        inp = torch.FloatTensor(noise).unsqueeze(0).to(DEVICE)
        out = model(inp).cpu().numpy()[0]

    mel_linear = librosa.db_to_power(out)
    inv_mel = librosa.filters.mel(sr=sr, n_fft=4096, n_mels=128)
    spec = np.dot(np.linalg.pinv(inv_mel), mel_linear)
    vocal = librosa.griffinlim(spec, n_iter=32, hop_length=256, n_fft=4096)

    # 길이 맞추기
    min_len = min(len(melody), len(vocal))
    melody = melody[:min_len]
    pad = pad[:min_len]
    vocal = vocal[:min_len]

    progress(0.7, desc="🎚️ 믹싱 중...")

    # 믹스
    vocal_norm = vocal / (np.max(np.abs(vocal)) + 1e-8) * 0.5
    mix = melody + pad + vocal_norm
    mix = mix / (np.max(np.abs(mix)) + 1e-8) * 0.9

    # 페이드 인/아웃
    fade = int(sr * 0.5)
    mix[:fade] *= np.linspace(0, 1, fade)
    mix[-fade:] *= np.linspace(1, 0, fade)

    progress(0.9, desc="💾 저장 중...")

    out_path = OUTPUT_DIR / f"demo_{style}_{datetime.now().strftime('%H%M%S')}.wav"
    sf.write(str(out_path), mix, sr)

    # 파형 이미지
    fig, ax = plt.subplots(figsize=(10, 3))
    times = np.arange(len(mix)) / sr
    ax.plot(times, mix, linewidth=0.3, color="#27AE60")
    ax.set_title(f"Demo: {style} ({duration_sec}s)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig_path = tempfile.mktemp(suffix=".png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    info = f"""### 🎵 데모곡 생성 완료

| 항목 | 값 |
|------|-----|
| **스타일** | {style} |
| **길이** | {duration_sec}초 |
| **BPM** | {params['bpm']} |
| **샘플레이트** | {sr:,} Hz |
| **저장 위치** | `{out_path}` |
"""
    return str(out_path), fig_path, info


# ─── Tab 5: Suno 곡 가져오기 ─────────────────────
# Suno 웹에서 생성한 곡을 URL로 가져와서 RVC 파이프라인에 연결


def suno_download_from_url(url: str, progress=gr.Progress()):
    """Suno 곡 URL에서 오디오를 다운로드합니다.
    
    지원 형식:
    - https://suno.com/song/UUID
    - https://cdn1.suno.ai/UUID.mp3
    - 직접 MP3 URL
    """
    import re
    url = url.strip()
    if not url:
        return None, None, "❌ URL을 입력해주세요."

    progress(0.1, desc="🔗 URL 분석 중...")

    # Suno 곡 페이지 URL → CDN URL 변환
    song_id_match = re.search(r'suno\.com/song/([a-f0-9-]+)', url)
    if song_id_match:
        song_id = song_id_match.group(1)
        audio_url = f"https://cdn1.suno.ai/{song_id}.mp3"
    elif re.match(r'https?://cdn\d*\.suno\.ai/.+\.mp3', url):
        audio_url = url
    elif url.startswith("http") and url.endswith(".mp3"):
        audio_url = url
    else:
        # Suno 페이지에서 오디오 URL 추출 시도
        song_id_match = re.search(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', url)
        if song_id_match:
            song_id = song_id_match.group(1)
            audio_url = f"https://cdn1.suno.ai/{song_id}.mp3"
        else:
            return None, None, "❌ 유효한 Suno URL이 아닙니다.\n\n지원 형식:\n- `https://suno.com/song/UUID`\n- `https://cdn1.suno.ai/UUID.mp3`"

    progress(0.3, desc="💾 오디오 다운로드 중...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # URL에서 song ID 추출
    id_match = re.search(r'([a-f0-9-]{36})', audio_url)
    short_id = id_match.group(1)[:8] if id_match else "suno"
    save_name = f"{timestamp}_{short_id}.mp3"
    save_path = SUNO_OUTPUT_DIR / save_name

    try:
        req = urllib.request.Request(audio_url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 1000:
            return None, None, f"❌ 다운로드된 파일이 너무 작습니다 ({len(data)} bytes). URL을 확인해주세요."
        save_path.write_bytes(data)
    except Exception as e:
        return None, None, f"❌ 다운로드 실패: {e}\nURL: {audio_url}"

    progress(0.9, desc="✅ 완료!")

    size_kb = save_path.stat().st_size / 1024
    status_md = f"""### ✅ Suno 곡 다운로드 완료

| 항목 | 값 |
|------|-----|
| **파일** | `{save_path}` |
| **크기** | {size_kb:.0f} KB |
| **원본 URL** | {audio_url} |

🎤 **RVC 음성 변환 탭**에서 위 파일로 음성 변환할 수 있습니다.
"""
    return str(save_path), str(save_path), status_md


def suno_list_outputs():
    """생성된 Suno 곡 목록"""
    files = sorted(SUNO_OUTPUT_DIR.glob("*.mp3"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return "아직 생성된 Suno 곡이 없습니다."
    lines = ["| 파일명 | 크기 | 생성일시 |", "|--------|------|---------|"]
    for f in files[:10]:
        size = f.stat().st_size / 1024
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M")
        lines.append(f"| `{f.name}` | {size:.0f} KB | {mtime} |")
    return "\n".join(lines)


# ─── 모델 목록 ───────────────────────────────────
def list_models():
    models = list(MODEL_DIR.glob("*.pth"))
    if not models:
        return "아직 학습된 모델이 없습니다."
    lines = []
    for m in sorted(models, key=lambda x: x.stat().st_mtime, reverse=True):
        size = m.stat().st_size / 1024 / 1024
        mtime = datetime.fromtimestamp(m.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"| `{m.name}` | {size:.1f} MB | {mtime} |")
    header = "| 모델 | 크기 | 학습일시 |\n|------|------|---------|"
    return header + "\n" + "\n".join(lines)


# ─── Gradio UI ────────────────────────────────────
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(theme=THEME, title="🎤 AI Singer Studio") as app:
    gr.Markdown("# 🎤 AI Singer Studio\n> KK의 목소리를 학습하고, AI가 노래하게 만드는 파이프라인")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 모델 목록")
            model_list = gr.Markdown(list_models())
            refresh_btn = gr.Button("🔄 새로고침", size="sm")
            refresh_btn.click(list_models, outputs=model_list)

        with gr.Column(scale=3):
            with gr.Tabs():
                # ── Tab 1: 보컬 업로드
                with gr.Tab("📁 보컬 업로드"):
                    gr.Markdown("WAV 파일을 업로드하면 자동으로 분석하고 학습 데이터로 등록합니다.")
                    upload_audio = gr.Audio(type="filepath", label="보컬 샘플 (WAV)")
                    analyze_btn = gr.Button("🔍 분석 시작", variant="primary")
                    with gr.Row():
                        spec_img = gr.Image(label="스펙트로그램")
                        analysis_info = gr.Markdown()
                    saved_path = gr.Textbox(visible=False)
                    analyze_btn.click(analyze_vocal, inputs=upload_audio, outputs=[spec_img, analysis_info, saved_path])

                # ── Tab 2: 모델 학습
                with gr.Tab("🔥 모델 학습"):
                    gr.Markdown("업로드된 보컬 샘플로 AI 보컬 모델을 학습합니다.")
                    with gr.Row():
                        epochs_slider = gr.Slider(10, 200, value=50, step=10, label="에포크 수")
                        lr_slider = gr.Slider(0.00001, 0.001, value=0.0001, step=0.00001, label="학습률")
                        quality_radio = gr.Radio(
                            ["기본 (80 mel)", "고품질 (128 mel)"],
                            value="고품질 (128 mel)", label="품질"
                        )
                    train_btn = gr.Button("🚀 학습 시작", variant="primary")
                    loss_img = gr.Image(label="학습 손실 그래프")
                    train_info = gr.Markdown()
                    train_btn.click(
                        train_model,
                        inputs=[epochs_slider, lr_slider, quality_radio],
                        outputs=[loss_img, train_info],
                    ).then(list_models, outputs=model_list)

                # ── Tab 3: 음성 변환
                with gr.Tab("🎤 음성 변환"):
                    gr.Markdown("학습된 모델로 오디오를 KK의 목소리로 변환합니다.")
                    convert_audio = gr.Audio(type="filepath", label="변환할 오디오")
                    pitch_slider = gr.Slider(-12, 12, value=0, step=1, label="피치 조절 (반음)")
                    convert_btn = gr.Button("🔄 변환 시작", variant="primary")
                    converted_audio = gr.Audio(label="변환 결과", type="filepath")
                    convert_info = gr.Markdown()
                    convert_btn.click(
                        convert_voice,
                        inputs=[convert_audio, pitch_slider],
                        outputs=[converted_audio, convert_info],
                    )

                # ── Tab 4: 데모곡 생성
                with gr.Tab("🎵 데모곡 생성"):
                    gr.Markdown("AI 보컬 + 악기 트랙을 합성해 데모곡을 만듭니다.")
                    with gr.Row():
                        style_dropdown = gr.Dropdown(
                            ["발라드", "팝", "일렉트로닉", "락"],
                            value="팝", label="스타일"
                        )
                        dur_slider = gr.Slider(5, 60, value=15, step=5, label="길이 (초)")
                    demo_btn = gr.Button("🎶 데모 생성", variant="primary")
                    demo_audio = gr.Audio(label="데모곡", type="filepath")
                    demo_img = gr.Image(label="파형")
                    demo_info = gr.Markdown()
                    demo_btn.click(
                        generate_demo,
                        inputs=[style_dropdown, dur_slider],
                        outputs=[demo_audio, demo_img, demo_info],
                    )

                # ── Tab 5: Suno 곡 가져오기
                with gr.Tab("🎹 Suno 곡 가져오기"):
                    gr.Markdown(
                        "### 🎹 Suno에서 만든 곡을 가져옵니다\n"
                        "> [suno.com](https://suno.com)에서 곡을 생성한 뒤, 곡 URL을 아래에 붙여넣으세요.\n"
                        "> 다운로드된 곡은 `outputs/suno/`에 저장되며 **RVC 음성 변환 탭**에서 바로 사용 가능합니다."
                    )

                    with gr.Row():
                        with gr.Column(scale=3):
                            suno_url_input = gr.Textbox(
                                label="🔗 Suno 곡 URL",
                                placeholder="https://suno.com/song/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                                lines=1,
                            )
                            gr.Markdown(
                                "**지원 형식:**\n"
                                "- `https://suno.com/song/UUID` — Suno 곡 페이지 URL\n"
                                "- `https://cdn1.suno.ai/UUID.mp3` — 직접 CDN URL\n"
                                "- 기타 MP3 직접 링크"
                            )
                        with gr.Column(scale=1):
                            gr.Markdown("#### 📂 다운로드된 Suno 곡")
                            suno_file_list = gr.Markdown(suno_list_outputs())
                            suno_refresh_btn = gr.Button("🔄 목록 새로고침", size="sm")

                    suno_dl_btn = gr.Button("📥 곡 다운로드", variant="primary", size="lg")

                    with gr.Row():
                        suno_audio_out = gr.Audio(
                            label="🔊 다운로드된 곡 (재생)",
                            type="filepath",
                        )
                        with gr.Column():
                            suno_status = gr.Markdown("*Suno 곡 URL을 입력하고 다운로드 버튼을 누르세요.*")

                    # RVC 탭 연결용 — 숨겨진 경로 출력
                    suno_filepath_out = gr.Textbox(
                        label="📁 파일 경로 (RVC 음성 변환에서 사용)",
                        interactive=False,
                        placeholder="다운로드 후 파일 경로가 여기 표시됩니다.",
                    )

                    # 이벤트 연결
                    suno_dl_btn.click(
                        suno_download_from_url,
                        inputs=[suno_url_input],
                        outputs=[suno_audio_out, suno_filepath_out, suno_status],
                    ).then(suno_list_outputs, outputs=suno_file_list)

                    suno_refresh_btn.click(suno_list_outputs, outputs=suno_file_list)

                    # RVC 탭의 convert_audio 입력에 파일 경로 직접 연결
                    suno_filepath_out.change(
                        fn=lambda p: p,
                        inputs=[suno_filepath_out],
                        outputs=[convert_audio],
                    )

    gr.Markdown("---\n*Built with 🔥 by KK & Chloe 🦞 — [GitHub](https://github.com/maker-KK/ai-singer)*")


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
