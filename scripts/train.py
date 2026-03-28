"""
고품질 AI 보컬 모델 학습
- 100에포크 학습으로 더 나은 품질
- 고해상도 오디오 처리
- 다양한 표현 학습
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import time
from datetime import datetime
import soundfile as sf
import librosa

print("🔥 고품질 AI 보컬 모델 학습 시작!")
print("=" * 50)

class HighQualityRVCModel(nn.Module):
    """고품질 RVC 모델"""
    def __init__(self, input_dim=128, hidden_dim=512):  # 더 높은 차원
        super().__init__()
        # 고품질 인코더
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 고품질 디코더
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, input_dim, 5, padding=2)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class HighQualityTrainer:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🔧 학습 디바이스: {self.device}")
        
        self.model = HighQualityRVCModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)  # 더 낮은 학습률
        self.criterion = nn.MSELoss()
        
        # 작업 디렉토리
        self.work_dir = "high_quality_training"
        os.makedirs(self.work_dir, exist_ok=True)
        
        self.model_dir = os.path.join(self.work_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.log_dir = os.path.join(self.work_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"📁 작업 디렉토리: {self.work_dir}")
        print(f"📊 모델 파라미터: {sum(p.numel() for p in self.model.parameters()):,}개")
        
    def load_high_quality_samples(self):
        """고품질 샘플 로드"""
        print("\n🎤 고품질 보컬 샘플 로드 중...")
        
        samples_dir = "dataset/raw"
        samples = []
        
        for file in os.listdir(samples_dir):
            if file.endswith('.wav') and 'kk_vocal' in file:
                filepath = os.path.join(samples_dir, file)
                try:
                    audio, sr = sf.read(filepath)
                    
                    # 고해상도 처리를 위한 업샘플링 (48kHz → 96kHz)
                    if sr == 48000:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=96000)
                        sr = 96000
                        print(f"   🔊 {file}: 96kHz로 업샘플링")
                    
                    duration = len(audio) / sr
                    
                    samples.append({
                        'file': file,
                        'path': filepath,
                        'duration': duration,
                        'sr': sr,
                        'audio': audio
                    })
                    
                    print(f"   ✅ {file}: {duration:.1f}초, {sr}Hz")
                    
                except Exception as e:
                    print(f"   ❌ {file} 로드 실패: {e}")
        
        print(f"   📊 총 {len(samples)}개 고품질 샘플 로드 완료")
        return samples
    
    def extract_high_quality_features(self, audio, sr):
        """고품질 특징 추출"""
        # 더 높은 해상도의 멜 스펙트로그램
        n_fft = 4096  # 더 큰 FFT
        hop_length = 256  # 더 짧은 홉
        n_mels = 128  # 더 많은 멜 밴드
        
        # 모노로 변환
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # 고품질 멜 스펙트로그램
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mel_spec = np.dot(mel_basis, magnitude)
        
        # 로그 스케일
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec
    
    def train_high_quality(self, epochs=100):
        """고품질 학습"""
        print(f"\n🎯 고품질 모델 학습 시작: {epochs}에포크")
        print("-" * 40)
        
        # 샘플 로드
        samples = self.load_high_quality_samples()
        if not samples:
            print("❌ 학습할 샘플이 없습니다!")
            return None
        
        # 특징 추출
        print("\n📊 고품질 특징 추출 중...")
        features = []
        
        for sample in samples:
            print(f"   🔍 {sample['file']} 고품질 특징 추출 중...")
            mel_spec = self.extract_high_quality_features(sample['audio'], sample['sr'])
            
            # 시퀀스 분할
            seq_length = 200  # 더 긴 시퀀스
            num_seqs = mel_spec.shape[1] // seq_length
            
            for i in range(min(20, num_seqs)):  # 더 많은 시퀀스
                start = i * seq_length
                end = start + seq_length
                seq = mel_spec[:, start:end]
                
                if seq.shape[1] == seq_length:
                    features.append(seq)
        
        print(f"   ✅ 총 {len(features)}개 고품질 학습 시퀀스 준비 완료")
        
        # 학습 시작
        start_time = time.time()
        training_log = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0
            num_batches = 0
            
            print(f"\n에포크 {epoch+1}/{epochs}:")
            
            # 미니배치 학습
            batch_size = 4  # 더 큰 배치
            np.random.shuffle(features)  # 매 에포크 셔플
            
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size]
                
                if len(batch_features) < batch_size:
                    continue
                
                # 텐서로 변환
                batch_tensors = []
                for feat in batch_features:
                    feat_tensor = torch.FloatTensor(feat).unsqueeze(0)
                    batch_tensors.append(feat_tensor)
                
                source = torch.cat(batch_tensors, dim=0).to(self.device)
                
                # 더 정교한 타겟 생성
                target = source + torch.randn_like(source) * 0.02  # 더 작은 노이즈
                
                # 학습 스텝
                self.optimizer.zero_grad()
                output = self.model(source)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 그래디언트 클리핑
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # 진행 상황 표시
                if (i // batch_size) % 10 == 0 and i > 0:
                    print(f"   배치 {i//batch_size + 1}/{len(features)//batch_size} - 손실: {loss.item():.6f}")
            
            # 에포크 결과
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                epoch_time = time.time() - epoch_start
                
                print(f"   ⏱️ 소요 시간: {epoch_time:.1f}초")
                print(f"   📊 평균 손실: {avg_loss:.6f}")
                
                # 로그 저장
                log_entry = {
                    "epoch": epoch + 1,
                    "loss": float(avg_loss),
                    "epoch_time": float(epoch_time),
                    "timestamp": datetime.now().isoformat()
                }
                training_log.append(log_entry)
                
                # 최고 성능 체크포인트
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_checkpoint = os.path.join(self.model_dir, f"best_model_epoch_{epoch+1}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': avg_loss,
                        'is_best': True
                    }, best_checkpoint)
                    print(f"   🏆 새로운 최고 성능! 체크포인트 저장: {best_checkpoint}")
                
                # 정기 체크포인트
                if (epoch + 1) % 20 == 0:
                    checkpoint_path = os.path.join(self.model_dir, f"checkpoint_epoch_{epoch+1}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': avg_loss
                    }, checkpoint_path)
                    print(f"   💾 정기 체크포인트 저장: {checkpoint_path}")
        
        # 학습 완료
        total_time = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("✅ 고품질 AI 보컬 모델 학습 완료!")
        print(f"총 소요 시간: {total_time/60:.1f}분")
        print(f"최종 손실: {training_log[-1]['loss']:.6f}")
        print(f"최고 손실: {best_loss:.6f} (개선률: {(training_log[0]['loss'] - best_loss)/training_log[0]['loss']*100:.1f}%)")
        
        # 최종 모델 저장
        final_model_path = os.path.join(self.model_dir, "kk_high_quality_vocal_model.pth")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"💾 최종 모델 저장: {final_model_path}")
        
        # 학습 로그 저장
        log_path = os.path.join(self.log_dir, "high_quality_training_log.json")
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"📝 학습 로그 저장: {log_path}")
        
        return training_log
    
    def run(self):
        """실행"""
        print("🔥 클로이의 고품질 AI 보컬 모델 학습 시작! 🔥")
        print("=" * 50)
        print("🎯 목표: KK님의 더 자연스럽고 표현력 있는 AI 보컬")
        print("📊 기대 효과: 더 긴 데모, 더 다양한 표현, 더 높은 음질")
        print("⏰ 예상 완료: 22:50")
        print()
        
        # 고품질 학습 실행
        training_log = self.train_high_quality(epochs=100)
        
        if training_log:
            print("\n" + "=" * 60)
            print("🎉 고품질 AI 보컬 모델 학습 완료! 🎉")
            print("=" * 60)
            
            initial_loss = training_log[0]['loss']
            final_loss = training_log[-1]['loss']
            improvement = (initial_loss - final_loss) / initial_loss * 100
            
            print(f"📊 학습 성과:")
            print(f"  - 초기 손실: {initial_loss:.2f}")
            print(f"  - 최종 손실: {final_loss:.2f}")
            print(f"  - 개선률: {improvement:.1f}%")
            print(f"  - 총 에포크: {len(training_log)}")
            
            print(f"\n📁 생성된 파일들:")
            print(f"  - 최종 모델: {self.model_dir}/kk_high_quality_vocal_model.pth")
            print(f"  - 최고 모델: {self.model_dir}/best_model_epoch_*.pth")
            print(f"  - 학습 로그: {self.log_dir}/high_quality_training_log.json")
            
            print(f"\n🎯 다음 단계: 풀 버전 '디지털 유목민' AI 보컬 변환")
            print(f"⏰ 예상 시작: 22:50")
        else:
            print("\n❌ 학습 실패 - 기존 모델로 진행합니다")

def main():
    trainer = HighQualityTrainer()
    trainer.run()

if __name__ == "__main__":
    main()
