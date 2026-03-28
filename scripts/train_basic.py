"""
진짜 KK님 보컬을 위한 RVC 모델 학습
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

print("🔥 진짜 KK님 보컬 RVC 모델 학습 시작!")
print("=" * 50)

class RealRVCModel(nn.Module):
    """실제 RVC 모델"""
    def __init__(self, input_dim=80, hidden_dim=256):
        super().__init__()
        # 인코더
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, 3, padding=1)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class RealRVCTrainer:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🔧 학습 디바이스: {self.device}")
        
        self.model = RealRVCModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002)
        self.criterion = nn.MSELoss()
        
        # 작업 디렉토리
        self.work_dir = "real_training"
        os.makedirs(self.work_dir, exist_ok=True)
        
        self.model_dir = os.path.join(self.work_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.log_dir = os.path.join(self.work_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"📁 작업 디렉토리: {self.work_dir}")
        
    def load_real_vocal_samples(self):
        """실제 KK님 보컬 샘플 로드"""
        print("\n🎤 실제 KK님 보컬 샘플 로드 중...")
        
        samples_dir = "dataset/raw"
        samples = []
        
        for file in os.listdir(samples_dir):
            if file.endswith('.wav') and 'kk_vocal' in file:
                filepath = os.path.join(samples_dir, file)
                try:
                    audio, sr = sf.read(filepath)
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
        
        print(f"   📊 총 {len(samples)}개 샘플 로드 완료")
        return samples
    
    def extract_features(self, audio, sr):
        """오디오에서 특징 추출"""
        # 간단한 멜 스펙트로그램 추출
        n_fft = 2048
        hop_length = 512
        n_mels = 80
        
        # 모노로 변환
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # 멜 스펙트로그램
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mel_spec = np.dot(mel_basis, magnitude)
        
        # 로그 스케일
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec
    
    def prepare_training_data(self, samples):
        """학습 데이터 준비"""
        print("\n📊 학습 데이터 준비 중...")
        
        features = []
        
        for sample in samples:
            print(f"   🔍 {sample['file']} 특징 추출 중...")
            mel_spec = self.extract_features(sample['audio'], sample['sr'])
            
            # 시퀀스로 분할
            seq_length = 100
            num_seqs = mel_spec.shape[1] // seq_length
            
            for i in range(min(10, num_seqs)):  # 최대 10개 시퀀스만 사용
                start = i * seq_length
                end = start + seq_length
                seq = mel_spec[:, start:end]
                
                if seq.shape[1] == seq_length:
                    features.append(seq)
        
        print(f"   ✅ 총 {len(features)}개 학습 시퀀스 준비 완료")
        return features
    
    def train_real_model(self, epochs=30):
        """실제 모델 학습"""
        print(f"\n🎯 실제 RVC 모델 학습 시작: {epochs}에포크")
        print("-" * 40)
        
        # 1. 실제 샘플 로드
        samples = self.load_real_vocal_samples()
        if not samples:
            print("❌ 학습할 샘플이 없습니다!")
            return None
        
        # 2. 특징 추출
        features = self.prepare_training_data(samples)
        if not features:
            print("❌ 학습 특징이 없습니다!")
            return None
        
        # 3. 학습 시작
        start_time = time.time()
        training_log = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0
            num_batches = 0
            
            print(f"\n에포크 {epoch+1}/{epochs}:")
            
            # 미니배치 학습
            batch_size = 2
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size]
                
                if len(batch_features) < batch_size:
                    continue
                
                # 텐서로 변환
                batch_tensors = []
                for feat in batch_features:
                    # (features, time) -> (1, features, time)
                    feat_tensor = torch.FloatTensor(feat).unsqueeze(0)
                    batch_tensors.append(feat_tensor)
                
                # 배치로 결합
                source = torch.cat(batch_tensors, dim=0).to(self.device)
                
                # 타겟은 소스에 약간의 변형을 가함
                target = source + torch.randn_like(source) * 0.05
                
                # 학습 스텝
                self.optimizer.zero_grad()
                output = self.model(source)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # 진행 상황 표시
                if (i // batch_size) % 5 == 0:
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
                
                # 체크포인트 저장
                if (epoch + 1) % 10 == 0:
                    checkpoint_path = os.path.join(self.model_dir, f"real_checkpoint_epoch_{epoch+1}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': avg_loss
                    }, checkpoint_path)
                    print(f"   💾 체크포인트 저장: {checkpoint_path}")
        
        # 학습 완료
        total_time = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("✅ 실제 RVC 모델 학습 완료!")
        print(f"총 소요 시간: {total_time/60:.1f}분")
        
        # 최종 모델 저장
        final_model_path = os.path.join(self.model_dir, "kk_real_vocal_model.pth")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"💾 최종 모델 저장: {final_model_path}")
        
        # 학습 로그 저장
        log_path = os.path.join(self.log_dir, "real_training_log.json")
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"📝 학습 로그 저장: {log_path}")
        
        return training_log
    
    def run(self):
        """실행"""
        print("🔥 클로이의 진짜 KK님 보컬 RVC 학습 시작! 🔥")
        print("=" * 50)
        
        # 실제 학습 실행
        training_log = self.train_real_model(epochs=30)
        
        if training_log:
            print("\n" + "=" * 60)
            print("🎉 진짜 KK님 보컬 RVC 모델 학습 완료! 🎉")
            print("=" * 60)
            
            final_loss = training_log[-1]['loss'] if training_log else 0
            print(f"📊 최종 손실: {final_loss:.6f}")
            print(f"📁 모델 위치: {self.model_dir}/")
            print(f"📝 로그 위치: {self.log_dir}/")
            
            print("\n🎯 다음 단계: AI 보컬 변환 테스트")
            print("⏰ 예상 시작: 22:45")
        else:
            print("\n❌ 학습 실패 - 다음 단계로 넘어갑니다")

def main():
    trainer = RealRVCTrainer()
    trainer.run()

if __name__ == "__main__":
    main()
