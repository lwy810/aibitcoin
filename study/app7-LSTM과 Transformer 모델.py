# 비트코인 5일 후 가격 예측 모델 비교
# LSTM과 Transformer 모델 비교 (5일 후 예측)
# btc_price_prediction_5days_lstm_transformer_pytorch

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 사용 가능한 한글 폰트 확인 및 설정
try:
    # Windows에서 사용 가능한 한글 폰트들
    korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic', 'Gulim']
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in korean_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"한글 폰트 설정: {font}")
            break
    else:
        print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        
except Exception as e:
    print(f"폰트 설정 중 오류: {e}")
    print("기본 폰트를 사용합니다.")

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

# 1. 데이터 로딩 및 전처리
print("데이터 로딩 중...")
df = yf.download('BTC-USD', start='2020-01-01', end='2024-12-31')[['Close']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def make_dataset(data, window_size=60, prediction_days=5):
    X, y = [], []
    # 5일 후 예측을 위해 범위를 조정 (prediction_days-1 만큼 줄임)
    for i in range(len(data) - window_size - prediction_days + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size+prediction_days-1])  # 5일 후 예측
    return np.array(X), np.array(y)

window_size = 60
X, y = make_dataset(scaled_data, window_size)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# PyTorch 텐서로 변환
X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# 2. LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 3. LSTM 모델 학습
print("LSTM 모델 학습 중...")
lstm_model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# 데이터로더 생성
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

lstm_model.train()
for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = lstm_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'LSTM Epoch [{epoch+1}/10], Loss: {total_loss/len(train_loader):.6f}')

# 4. LSTM 예측 및 시각화
lstm_model.eval()
with torch.no_grad():
    predicted_lstm = lstm_model(X_test)
    predicted_lstm = predicted_lstm.cpu().numpy()
    predicted_lstm = scaler.inverse_transform(predicted_lstm)
    actual = scaler.inverse_transform(y_test.cpu().numpy())

plt.figure(figsize=(12, 6))
plt.plot(actual, label='실제 가격', color='blue')
plt.plot(predicted_lstm, label='LSTM 예측', color='red')
plt.title("LSTM 기반 비트코인 5일 후 가격 예측")
plt.xlabel('시간')
plt.ylabel('가격 (USD)')
plt.legend()
plt.grid(True)
plt.show()

# 5. Transformer 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=2, num_layers=2):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = self.transformer(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x)  # (batch, d_model, 1)
        x = x.squeeze(-1)  # (batch, d_model)
        x = self.fc(x)
        return x

# 6. Transformer 모델 학습
print("Transformer 모델 학습 중...")
transformer_model = TransformerModel().to(device)
transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)

transformer_model.train()
for epoch in range(10):
    total_loss = 0
    for batch_x, batch_y in train_loader:
        transformer_optimizer.zero_grad()
        outputs = transformer_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        transformer_optimizer.step()
        total_loss += loss.item()
    print(f'Transformer Epoch [{epoch+1}/10], Loss: {total_loss/len(train_loader):.6f}')

# 7. Transformer 예측 및 시각화
transformer_model.eval()
with torch.no_grad():
    predicted_transformer = transformer_model(X_test)
    predicted_transformer = predicted_transformer.cpu().numpy()
    predicted_transformer = scaler.inverse_transform(predicted_transformer)

plt.figure(figsize=(12, 6))
plt.plot(actual, label='실제 가격', color='blue')
plt.plot(predicted_transformer, label='Transformer 예측', color='green')
plt.title("Transformer 기반 비트코인 5일 후 가격 예측")
plt.xlabel('시간')
plt.ylabel('가격 (USD)')
plt.legend()
plt.grid(True)
plt.show()

# 8. 두 모델 비교
plt.figure(figsize=(15, 8))
plt.plot(actual, label='실제 가격', color='blue', linewidth=2)
plt.plot(predicted_lstm, label='LSTM 예측', color='red', linewidth=1.5)
plt.plot(predicted_transformer, label='Transformer 예측', color='green', linewidth=1.5)
plt.title("LSTM vs Transformer: 비트코인 5일 후 가격 예측 비교")
plt.xlabel('시간')
plt.ylabel('가격 (USD)')
plt.legend()
plt.grid(True)
plt.show()

# 9. 성능 평가
from sklearn.metrics import mean_squared_error, mean_absolute_error

lstm_mse = mean_squared_error(actual, predicted_lstm)
lstm_mae = mean_absolute_error(actual, predicted_lstm)

transformer_mse = mean_squared_error(actual, predicted_transformer)
transformer_mae = mean_absolute_error(actual, predicted_transformer)

print("\n=== 모델 성능 비교 ===")
print(f"LSTM - MSE: {lstm_mse:.2f}, MAE: {lstm_mae:.2f}")
print(f"Transformer - MSE: {transformer_mse:.2f}, MAE: {transformer_mae:.2f}")

if lstm_mse < transformer_mse:
    print("LSTM 모델이 더 좋은 성능을 보입니다.")
else:
    print("Transformer 모델이 더 좋은 성능을 보입니다.") 