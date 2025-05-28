# 비트코인 Transformer 모델 백테스트 (5년간)
# 년수익률, MDD, 샤프지수, 승률 등 성과 분석

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
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

try:
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

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

# 1. 데이터 로딩 및 전처리
print("\n=== 데이터 로딩 중 ===")
df = yf.download('XRP-USD', start='2020-01-01', end='2024-12-31')[['Close']]
print(f"총 데이터 개수: {len(df)}일")

if len(df) > 0:
    min_price = float(df['Close'].min())
    max_price = float(df['Close'].max())
    print(f"데이터 범위: ${min_price:.2f} ~ ${max_price:.2f}")
else:
    print("데이터가 없습니다. 프로그램을 종료합니다.")
    exit()

# 원본 데이터 저장 (백테스트용)
original_df = df.copy()
original_df['Date'] = original_df.index

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

def make_dataset(data, window_size=60, prediction_days=1):
    X, y = [], []
    for i in range(len(data) - window_size - prediction_days + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size+prediction_days-1])  # +1일 예측
    return np.array(X), np.array(y)

# 2. Transformer 모델 정의
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
        x = x.transpose(1, 2)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

# 3. 백테스트를 위한 시계열 분할
print("\n=== 백테스트 데이터 준비 ===")
window_size = 60
prediction_days = 5  # +1일 예측으로 변경

# 전체 데이터셋 생성
X, y = make_dataset(scaled_data, window_size, prediction_days)

# 백테스트를 위한 시간 순서 유지
total_samples = len(X)
train_size = int(total_samples * 0.7)  # 70% 학습용
test_size = total_samples - train_size  # 30% 백테스트용

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

print(f"학습 데이터: {len(X_train)}개")
print(f"백테스트 데이터: {len(X_test)}개")

# PyTorch 텐서로 변환
X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# 4. 모델 학습
print("\n=== Transformer 모델 학습 중 ===")
model = TransformerModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model.train()
for epoch in range(20):  # 에포크 수 증가
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/20], Loss: {total_loss/len(train_loader):.6f}')

# 5. 예측 및 백테스트
print("\n=== 백테스트 실행 중 ===")
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = predictions.cpu().numpy()
    predictions = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test.cpu().numpy())

# 백테스트 날짜 계산
start_backtest_idx = train_size + window_size + prediction_days - 1
backtest_dates = original_df.index[start_backtest_idx:start_backtest_idx + len(predictions)]

# 6. 트레이딩 시뮬레이션
print("\n=== 트레이딩 시뮬레이션 ===")
initial_capital = 100000  # 초기 자본 10만 달러
capital = initial_capital
position = 0  # 0: 현금, 1: 비트코인 보유
trades = []
portfolio_values = []
returns = []

# 현재 가격 (예측 시점의 실제 가격)
current_prices = []
for i in range(len(predictions)):
    if i < len(actual_prices) - prediction_days:
        # 예측 시점에서의 실제 가격 (1일 전 가격)
        current_price_idx = start_backtest_idx + i - prediction_days
        if current_price_idx >= 0:
            current_prices.append(original_df.iloc[current_price_idx]['Close'])
        else:
            current_prices.append(original_df.iloc[0]['Close'])
    else:
        current_prices.append(actual_prices[i])

for i in range(len(predictions)):
    current_price = float(current_prices[i])
    predicted_price = float(predictions[i][0])
    actual_future_price = float(actual_prices[i][0])
    
    # 트레이딩 로직: 1일 후 예측 기반 거래 (+1일 예측에 맞게 조정)
    price_change_prediction = (predicted_price - current_price) / current_price
    
    if position == 0 and price_change_prediction > 0.01:  # 매수 신호: 1% 이상 상승 예상
        position = 1
        btc_amount = capital / current_price
        trades.append({
            'date': backtest_dates[i],
            'action': 'BUY',
            'price': current_price,
            'amount': btc_amount,
            'capital': capital
        })
    elif position == 1 and price_change_prediction < -0.005:  # 매도 신호: 0.5% 하락 예상
        position = 0
        capital = btc_amount * current_price
        trades.append({
            'date': backtest_dates[i],
            'action': 'SELL',
            'price': current_price,
            'amount': btc_amount,
            'capital': capital
        })
    
    # 포트폴리오 가치 계산
    if position == 1:
        portfolio_value = btc_amount * current_price
    else:
        portfolio_value = capital
    
    portfolio_values.append(portfolio_value)
    
    # 일일 수익률 계산
    if i > 0:
        daily_return = (portfolio_value - portfolio_values[i-1]) / portfolio_values[i-1]
        returns.append(daily_return)

# 마지막 포지션 정리
if position == 1:
    final_capital = float(btc_amount * current_prices[-1])
else:
    final_capital = float(capital)

# 7. 성과 분석
print("\n=== 백테스트 성과 분석 ===")

# 총 수익률
total_return = (final_capital - initial_capital) / initial_capital * 100

# 연수익률 (CAGR)
years = len(predictions) / 365
cagr = (final_capital / initial_capital) ** (1/years) - 1

# 최대 낙폭 (MDD) 계산
peak = initial_capital
max_drawdown = 0
for value in portfolio_values:
    if value > peak:
        peak = value
    drawdown = (peak - value) / peak
    if drawdown > max_drawdown:
        max_drawdown = drawdown

# 샤프 지수 계산
if len(returns) > 0:
    returns_array = np.array(returns)
    sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
else:
    sharpe_ratio = 0

# 승률 계산
winning_trades = 0
total_trades = len([t for t in trades if t['action'] == 'SELL'])

for i in range(len(trades)-1):
    if trades[i]['action'] == 'BUY' and trades[i+1]['action'] == 'SELL':
        if trades[i+1]['capital'] > trades[i]['capital']:
            winning_trades += 1

win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

# Buy & Hold 전략과 비교
buy_hold_return = float((current_prices[-1] - current_prices[0]) / current_prices[0] * 100)

print(f"📊 백테스트 기간: {backtest_dates[0].strftime('%Y-%m-%d')} ~ {backtest_dates[-1].strftime('%Y-%m-%d')}")
print(f"📈 초기 자본: ${initial_capital:,.2f}")
print(f"💰 최종 자본: ${final_capital:,.2f}")
print(f"📊 총 수익률: {total_return:.2f}%")
print(f"📈 연수익률 (CAGR): {cagr*100:.2f}%")
print(f"📉 최대 낙폭 (MDD): {max_drawdown*100:.2f}%")
print(f"⚡ 샤프 지수: {sharpe_ratio:.3f}")
print(f"🎯 승률: {win_rate:.1f}%")
print(f"🔄 총 거래 횟수: {total_trades}회")
print(f"📊 Buy & Hold 수익률: {buy_hold_return:.2f}%")

# 8. 시각화
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1) 포트폴리오 가치 변화
ax1.plot(backtest_dates, portfolio_values, label='AI 전략', color='blue', linewidth=2)
buy_hold_values = [initial_capital * (price / current_prices[0]) for price in current_prices]
ax1.plot(backtest_dates, buy_hold_values, label='Buy & Hold', color='red', linewidth=2)
ax1.set_title('포트폴리오 가치 변화')
ax1.set_ylabel('포트폴리오 가치 ($)')
ax1.legend()
ax1.grid(True)

# 2) 예측 vs 실제 가격
ax2.plot(backtest_dates, actual_prices.flatten(), label='실제 가격', color='blue', alpha=0.7)
ax2.plot(backtest_dates, predictions.flatten(), label='예측 가격', color='red', alpha=0.7)
ax2.set_title('+1일 후 가격 예측 vs 실제')
ax2.set_ylabel('비트코인 가격 ($)')
ax2.legend()
ax2.grid(True)

# 3) 누적 수익률
cumulative_returns = [(v / initial_capital - 1) * 100 for v in portfolio_values]
buy_hold_cumulative = [(v / initial_capital - 1) * 100 for v in buy_hold_values]
ax3.plot(backtest_dates, cumulative_returns, label='AI 전략', color='blue', linewidth=2)
ax3.plot(backtest_dates, buy_hold_cumulative, label='Buy & Hold', color='red', linewidth=2)
ax3.set_title('누적 수익률 비교')
ax3.set_ylabel('누적 수익률 (%)')
ax3.legend()
ax3.grid(True)

# 4) 드로우다운
running_max = np.maximum.accumulate(portfolio_values)
drawdowns = [(running_max[i] - portfolio_values[i]) / running_max[i] * 100 for i in range(len(portfolio_values))]
ax4.fill_between(backtest_dates, 0, drawdowns, color='red', alpha=0.3)
ax4.set_title(f'드로우다운 (최대: {max_drawdown*100:.2f}%)')
ax4.set_ylabel('드로우다운 (%)')
ax4.grid(True)

plt.tight_layout()
plt.show()

# 9. 거래 내역 출력
print("\n=== 주요 거래 내역 ===")
for i, trade in enumerate(trades[:10]):  # 처음 10개 거래만 출력
    print(f"{trade['date'].strftime('%Y-%m-%d')} | {trade['action']} | ${trade['price']:.2f} | 자본: ${trade['capital']:.2f}")

if len(trades) > 10:
    print(f"... 총 {len(trades)}개 거래")

print("\n=== 백테스트 완료 ===")
