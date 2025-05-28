# 테더 코인 Transformer 모델 백테스트 (5년간)
# 년수익률, MDD, 샤프지수, 승률 등 성과 분석

import pyupbit
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
print("\n=== 테더 코인 데이터 로딩 중 ===")

# 테더 코인 5년간 일봉 데이터 가져오기
try:
    df = pyupbit.get_ohlcv('KRW-USDT', interval="day", count=2000)
    
    if df is None or df.empty:
        print("테더 코인 데이터를 가져올 수 없습니다.")
        exit()
    
    # close 컬럼만 사용하고 DataFrame 형태로 변환
    df = df[['close']].copy()
    df.columns = ['Close']  # 컬럼명 통일
    
    print(f"총 데이터 개수: {len(df)}일")
    print(f"데이터 기간: {df.index[0]} ~ {df.index[-1]}")
    
    min_price = float(df['Close'].min())
    max_price = float(df['Close'].max())
    print(f"가격 범위: {min_price:,.0f}원 ~ {max_price:,.0f}원")
    
except Exception as e:
    print(f"데이터 로딩 중 오류: {e}")
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
test_days = 365  # 1년간 테스트 기간

# 테스트 기간이 전체 데이터보다 클 경우 조정
if test_days >= total_samples:
    test_days = max(180, int(total_samples * 0.5))  # 최소 6개월 또는 50%
    print(f"데이터 부족으로 테스트 기간을 {test_days}일로 조정합니다.")

train_size = total_samples - test_days

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

print(f"학습 데이터: {len(X_train)}개")
print(f"백테스트 데이터: {len(X_test)}개 (약 {len(X_test)/365:.1f}년)")

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

# 예측값 분석 및 상대적 임계값 설정
price_changes = []
for i in range(len(predictions)):
    if i < len(actual_prices):
        current_price = float(actual_prices[i][0])
        predicted_price = float(predictions[i][0])
        price_change = (predicted_price - current_price) / current_price
        price_changes.append(price_change)

print(f"예측 변화율 범위: {min(price_changes)*100:.3f}% ~ {max(price_changes)*100:.3f}%")
print(f"예측 변화율 평균: {np.mean(price_changes)*100:.3f}%")
print(f"예측 변화율 표준편차: {np.std(price_changes)*100:.3f}%")

# 상대적 임계값 설정 (상위 30%, 하위 30% 기준)
buy_threshold = np.percentile(price_changes, 70)  # 상위 30% (상대적으로 덜 하락하는 구간)
sell_threshold = np.percentile(price_changes, 30)  # 하위 30% (상대적으로 더 하락하는 구간)

print(f"매수 임계값 (상위 30%): {buy_threshold*100:.3f}%")
print(f"매도 임계값 (하위 30%): {sell_threshold*100:.3f}%")

initial_capital = 100000  # 초기 자본 10만원
capital = initial_capital
position = 0  # 0: 현금, 1: 테더 보유
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
        current_prices.append(actual_prices[i][0])

for i in range(len(predictions)):
    current_price = float(current_prices[i])
    predicted_price = float(predictions[i][0])
    actual_future_price = float(actual_prices[i][0])
    
    # 트레이딩 로직: 5일 후 예측 기반 거래 (더 적극적인 조건)
    price_change_prediction = (predicted_price - current_price) / current_price
    
    # 예측 변화율의 상위/하위 30% 기준으로 거래
    if position == 0 and price_change_prediction > buy_threshold:  # 매수 신호: 상위 30% 이상 상승 예상
        position = 1
        usdt_amount = capital / current_price
        trades.append({
            'date': backtest_dates[i],
            'action': 'BUY',
            'price': current_price,
            'amount': usdt_amount,
            'capital': capital
        })
    elif position == 1 and price_change_prediction < sell_threshold:  # 매도 신호: 하위 30% 이하 하락 예상
        position = 0
        capital = usdt_amount * current_price
        trades.append({
            'date': backtest_dates[i],
            'action': 'SELL',
            'price': current_price,
            'amount': usdt_amount,
            'capital': capital
        })
    
    # 포트폴리오 가치 계산
    if position == 1:
        portfolio_value = usdt_amount * current_price
    else:
        portfolio_value = capital
    
    portfolio_values.append(portfolio_value)
    
    # 일일 수익률 계산
    if i > 0:
        daily_return = (portfolio_value - portfolio_values[i-1]) / portfolio_values[i-1]
        returns.append(daily_return)

# 마지막 포지션 정리
if position == 1:
    final_capital = float(usdt_amount * current_prices[-1])
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
print(f"📈 초기 자본: {initial_capital:,.0f}원")
print(f"💰 최종 자본: {final_capital:,.0f}원")
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

# Buy & Hold 값 계산 시 형태 문제 해결
try:
    buy_hold_values = []
    initial_price = float(current_prices[0])
    for price in current_prices:
        buy_hold_value = initial_capital * (float(price) / initial_price)
        buy_hold_values.append(buy_hold_value)
    
    ax1.plot(backtest_dates, buy_hold_values, label='Buy & Hold', color='red', linewidth=2)
except Exception as e:
    print(f"Buy & Hold 차트 그리기 오류: {e}")
    # Buy & Hold 없이 AI 전략만 표시
    pass

ax1.set_title('포트폴리오 가치 변화')
ax1.set_ylabel('포트폴리오 가치 (원)')
ax1.legend()
ax1.grid(True)

# 2) 예측 vs 실제 가격
ax2.plot(backtest_dates, actual_prices.flatten(), label='실제 가격', color='blue', alpha=0.7)
ax2.plot(backtest_dates, predictions.flatten(), label='예측 가격', color='red', alpha=0.7)
ax2.set_title('+5일 후 테더 코인 가격 예측 vs 실제')
ax2.set_ylabel('테더 코인 가격 (원)')
ax2.legend()
ax2.grid(True)

# 3) 누적 수익률
cumulative_returns = [(v / initial_capital - 1) * 100 for v in portfolio_values]

try:
    buy_hold_cumulative = []
    for buy_hold_value in buy_hold_values:
        cumulative_return = (buy_hold_value / initial_capital - 1) * 100
        buy_hold_cumulative.append(cumulative_return)
    
    ax3.plot(backtest_dates, cumulative_returns, label='AI 전략', color='blue', linewidth=2)
    ax3.plot(backtest_dates, buy_hold_cumulative, label='Buy & Hold', color='red', linewidth=2)
except Exception as e:
    print(f"누적 수익률 차트 그리기 오류: {e}")
    # AI 전략만 표시
    ax3.plot(backtest_dates, cumulative_returns, label='AI 전략', color='blue', linewidth=2)

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
    print(f"{trade['date'].strftime('%Y-%m-%d')} | {trade['action']} | {trade['price']:,.0f}원 | 자본: {trade['capital']:,.0f}원")

if len(trades) > 10:
    print(f"... 총 {len(trades)}개 거래")

print("\n=== 백테스트 완료 ===")
