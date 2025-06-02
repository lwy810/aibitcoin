import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit # For conceptual rolling window/cross-validation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- 1. 데이터 수집 ---
def get_bitcoin_data(start_date='2018-01-01', end_date='2024-01-01'):
    """Yahoo Finance에서 비트코인(BTC-USD) 과거 데이터를 가져옵니다."""
    print("비트코인 데이터 가져오는 중...")
    df = yf.download('BTC-USD', start=start_date, end=end_date)
    
    # MultiIndex가 있는 경우 평평하게 만들기
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    df = df[['Close', 'Volume']]
    print(f"비트코인 데이터 가져오기 완료. 형태: {df.shape}")
    return df

def get_google_trends_data(keyword='bitcoin', start_date='2018-01-01', end_date='2024-01-01'):
    """주어진 키워드에 대한 Google 트렌드 데이터를 가져옵니다."""
    print(f"'{keyword}'에 대한 Google 트렌드 데이터 가져오는 중...")
    pytrend = TrendReq(hl='en-US', tz=360) # tz=360은 미국 중부 시간을 나타냅니다.
    kw_list = [keyword]
    try:
        # Pytrends는 사용자 지정 범위에 대해 특정 날짜 형식을 요구합니다.
        # 일일 데이터의 경우, 광범위한 범위를 가져온 다음 필터링하는 것이 더 쉽습니다.
        timeframe = f'{start_date} {end_date}'
        pytrend.build_payload(kw_list, cat=0, timeframe=timeframe, geo='', gprop='')
        df_trends = pytrend.interest_over_time()
        if not df_trends.empty:
            # isPartial 컬럼이 있다면 제거
            if 'isPartial' in df_trends.columns:
                df_trends = df_trends.drop(columns=['isPartial'])
            df_trends = df_trends[[keyword]].rename(columns={keyword: 'GoogleTrends'})
            # 7일 이동 평균 계산 [1]
            df_trends = df_trends.rolling(window=7).mean()
            print(f"Google 트렌드 데이터 가져오기 완료. 형태: {df_trends.shape}")
            return df_trends
        else:
            print("지정된 기간에 대한 Google 트렌드 데이터가 없습니다.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Google 트렌드 데이터 가져오기 오류: {e}")
        # 빈 DataFrame 반환하여 코드가 계속 실행되도록 함
        return pd.DataFrame()

# --- 2. 특징 공학 및 전처리 ---
def calculate_technical_indicators(df):
    """주어진 DataFrame에 대해 RSI 및 MACD를 계산합니다."""
    print("기술 지표 계산 중...")
    # RSI (14일 기간) [1]
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12일 EMA, 26일 EMA, 신호선에 대한 9일 EMA) [1]
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - macd_signal

    # 평활화 기법: SMA 및 볼린저 밴드 (여기서는 SMA만 간단히 추가) [1]
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    print("기술 지표 계산 완료.")
    return df

def preprocess_data(df_btc, df_trends):
    """
    데이터를 통합하고, 결측치를 처리하고, 특징을 표준화합니다.
    df_btc와 df_trends가 'Date'를 인덱스로 가지고 있다고 가정합니다.
    """
    print("데이터 전처리 시작...")
    
    # Google Trends 데이터가 비어있는 경우 더미 데이터 생성
    if df_trends.empty:
        print("Google Trends 데이터가 없으므로 더미 데이터를 생성합니다.")
        df_trends = pd.DataFrame(index=df_btc.index)
        df_trends['GoogleTrends'] = 50  # 중간값으로 더미 데이터 채우기
    
    # 날짜 인덱스를 datetime 객체로 변환하여 적절한 병합 보장
    df_btc.index = pd.to_datetime(df_btc.index)
    df_trends.index = pd.to_datetime(df_trends.index)
    
    # MultiIndex가 있는 경우 평평하게 만들기
    if isinstance(df_btc.index, pd.MultiIndex):
        df_btc = df_btc.reset_index(level=1, drop=True)
    if isinstance(df_trends.index, pd.MultiIndex):
        df_trends = df_trends.reset_index(level=1, drop=True)

    # 날짜 인덱스로 DataFrame 정렬
    df = pd.merge(df_btc, df_trends, left_index=True, right_index=True, how='inner')

    # 기술 지표 계산
    df = calculate_technical_indicators(df)

    # 결측치 순방향 채우기 (forward-fill) [1]
    df = df.ffill()
    df = df.bfill() # 초기 NaN 값 처리

    # 랜덤 포레스트를 위한 목표 변수 생성: 다음 날 가격 움직임 (증가 시 1, 그렇지 않으면 0) [1]
    df['NextDayPriceIncrease'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # 마지막 행은 NextDayPriceIncrease가 NaN이 되므로 제거
    df.dropna(inplace=True)

    # 모델을 위한 특징 선택
    # 보고서는 기술 지표, 시장 심리, 거시 경제 변수, RF 예측을 언급합니다. [1]
    # 이 예시에서는 Close, Volume, RSI, MACD_Hist, GoogleTrends를 사용합니다.
    # 거시 경제 변수는 보고서에 상세히 설명되어 있지 않으므로 간단화를 위해 생략합니다.
    features = ['Close', 'Volume', 'RSI', 'MACD_Hist', 'GoogleTrends']
    X = df[features]
    y = df['NextDayPriceIncrease']

    # 수치 특징 표준화 [1] (신경망은 스케일에 민감하므로 중요)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df.index)

    print("데이터 전처리 완료.")
    return X_scaled_df, y, scaler, df # 나중에 신호 생성을 위해 원본 df 반환

def create_sequences(data, target, sequence_length=10):
    """
    롤링 윈도우를 기반으로 LSTM/GRU 모델을 위한 시퀀스를 생성합니다.
    [1] "모델은 이전 10일간의 비트코인 가격 데이터를 기반으로 훈련됩니다."
    """
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data.iloc[i:(i + sequence_length)].values
        y = target.iloc[i + sequence_length] # 시퀀스 다음 날을 예측
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- 3. 예측 분석을 위한 랜덤 포레스트 분류기 ---
class RandomForestPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42) # n_estimators는 튜닝 가능

    def train(self, X_train, y_train):
        print("랜덤 포레스트 분류기 훈련 중...")
        self.model.fit(X_train, y_train)
        print("랜덤 포레스트 분류기 훈련 완료.")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1] # 가격 상승 확률 (클래스 1)

# --- 4. 신경망 모델 (PyTorch) ---
class FNN(nn.Module):
    """보고서에 따른 피드포워드 신경망 (FNN)."""
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]): # 예시 hidden_dims, 보고서는 32-128 [1]
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dims[2], 1)
        self.sigmoid = nn.Sigmoid() # 출력층은 확률적 예측을 위해 시그모이드 사용 [1]

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        x = self.sigmoid(self.output_layer(x))
        return x

class LSTMModel(nn.Module):
    """보고서에 따른 장단기 기억 (LSTM) 네트워크."""
    def __init__(self, input_dim, hidden_dim=50, num_layers=2, dropout_rate=0.2): # 층당 50개 뉴런, 2개 층 [1]
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate) # 드롭아웃 [1]
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid() # 확률적 출력을 위한 시그모이드 가정

    def forward(self, x):
        # x 형태: (batch_size, sequence_length, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # 마지막 시간 단계의 출력 가져오기
        out = self.sigmoid(out)
        return out

class GRUModel(nn.Module):
    """보고서에 따른 게이트 순환 유닛 (GRU) 네트워크."""
    def __init__(self, input_dim, hidden_dim=40, num_layers=2): # 층당 40개 뉴런, 2개 층 [1]
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid() # 확률적 출력을 위한 시그모이드 가정

    def forward(self, x):
        # x 형태: (batch_size, sequence_length, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :]) # 마지막 시간 단계의 출력 가져오기
        out = self.sigmoid(out)
        return out

# --- 5. 앙상블 모델 ---
class EnsembleModel(nn.Module):
    """FNN, LSTM, GRU의 예측을 지정된 가중치로 결합합니다."""
    def __init__(self, fnn_model, lstm_model, gru_model, weights={'fnn': 0.4, 'lstm': 0.3, 'gru': 0.3}): # 가중치 [1]
        super(EnsembleModel, self).__init__()
        self.fnn = fnn_model
        self.lstm = lstm_model
        self.gru = gru_model
        self.weights = weights

    def forward(self, fnn_input, seq_input):
        fnn_pred = self.fnn(fnn_input)
        lstm_pred = self.lstm(seq_input)
        gru_pred = self.gru(seq_input)

        # 모든 예측이 2D 텐서 (batch_size, 1)인지 확인
        fnn_pred = fnn_pred.view(-1, 1)
        lstm_pred = lstm_pred.view(-1, 1)
        gru_pred = gru_pred.view(-1, 1)

        # 예측의 가중치 합
        ensemble_pred = (self.weights['fnn'] * fnn_pred +
                         self.weights['lstm'] * lstm_pred +
                         self.weights['gru'] * gru_pred)
        return ensemble_pred

# --- 6. 훈련 함수 ---
def train_pytorch_model(model, dataloader, criterion, optimizer, epochs=200, model_name="Model"):
    """PyTorch 모델을 훈련합니다."""
    print(f"{model_name} {epochs} 에포크 동안 훈련 중...")
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.float().view_as(output))
            loss.backward()
            optimizer.step()
        # if (epoch + 1) % 20 == 0: # 진행 상황을 너무 자주 출력하지 않도록 주석 처리
        #     print(f'  {model_name} Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    print(f"{model_name} 훈련 완료.")

def evaluate_pytorch_model(model, dataloader, model_name="Model"):
    """PyTorch 모델을 평가하고 예측을 반환합니다."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for data, _ in dataloader:
            output = model(data)
            predictions.extend(output.cpu().numpy())
    return np.array(predictions).flatten()

# --- 7. 거래 신호 생성 ---
def generate_trading_signal(rsi_val, macd_hist_val, google_trends_scaled_val, ml_ensemble_proba):
    """
    가중치 점수를 계산하고 매수/매도/보유 신호를 결정합니다.
    각 개별 신호는 일반적으로 -1에서 1 사이의 범위로 정규화/매핑됩니다.
    여기서 1은 강한 매수 경향을, -1은 강한 매도 경향을 나타냅니다.
    """

    # RSI 신호: 낮은 RSI(과매도)는 강세, 높은 RSI(과매수)는 약세입니다.
    # RSI(0-100)를 -1에서 1 사이의 신호로 매핑합니다.
    if rsi_val < 30:
        rsi_signal = 1.0 # 강한 매수
    elif rsi_val > 70:
        rsi_signal = -1.0 # 강한 매도
    else:
        # 30에서 70 사이의 선형 보간, 1에서 -1로 매핑
        rsi_signal = 1 - 2 * ((rsi_val - 30) / 40)
        rsi_signal = np.clip(rsi_signal, -1.0, 1.0) # -1과 1 사이에 유지되도록 클리핑

    # MACD 신호: MACD 히스토그램 (MACD - MACD_Signal). 양수는 강세, 음수는 약세입니다.
    # 보고서에 MACD 신호의 정확한 정규화가 명시되어 있지 않으므로, 간단한 부호 기반 접근 방식을 사용합니다.
    if macd_hist_val > 0:
        macd_signal = 1.0
    elif macd_hist_val < 0:
        macd_signal = -1.0
    else:
        macd_signal = 0.0

    # Google 트렌드 신호: 스케일링된 값 (StandardScaler에서, 0을 중심으로).
    # 스케일링된 값이 높을수록 (더 많은 관심) 강세, 낮을수록 약세입니다.
    # 스케일링된 값을 -1에서 1로 매핑합니다.
    # 대부분의 값이 +/- 2 표준 편차 내에 있다고 가정하고 선형적으로 매핑합니다.
    google_trends_signal = np.clip(google_trends_scaled_val / 2.0, -1.0, 1.0)

    # ML 신호: 앙상블에서 나온 확률 (0-1). 가격 상승 확률이 높을수록 강세입니다.
    # 0-1 확률을 -1에서 1 신호로 매핑합니다. (0 -> -1, 0.5 -> 0, 1 -> 1)
    ml_signal = (ml_ensemble_proba * 2) - 1
    ml_signal = np.clip(ml_signal, -1.0, 1.0) # -1과 1 사이에 유지되도록 클리핑

    # 가중치 점수 계산 [1]
    weighted_score = (0.2 * rsi_signal +
                      0.4 * macd_signal +
                      0.2 * google_trends_signal +
                      0.4 * ml_signal)

    # 가중치 점수를 기반으로 한 거래 결정 [1]
    if weighted_score > 0.5:
        return "BUY", weighted_score
    elif weighted_score < -0.5:
        return "SELL", weighted_score
    else:
        return "HOLD", weighted_score

# --- 8. 백테스팅 ---
def backtest_strategy(df, signals_df, initial_capital=10000, stop_loss_pct=0.10):
    """
    거래 전략의 간소화된 백테스트를 수행합니다.
    보고서의 간소화에 따라 거래 비용, 슬리피지 또는 세금은 고려하지 않습니다. [1]
    """
    print(f"백테스팅 시작 (손절매: {stop_loss_pct*100}%)")
    capital = initial_capital
    btc_holdings = 0
    entry_price = 0 # 진입 가격 추적
    portfolio_value = []
    trade_log = []

    # 원본 데이터와 신호 병합
    df_merged = df.copy()
    # signals_df의 인덱스가 original_df의 서브셋이므로, original_df를 기준으로 정렬
    df_merged = df_merged.loc[signals_df.index]
    df_merged['Signal'] = signals_df['Signal']
    df_merged['Score'] = signals_df['Score']

    for i in range(len(df_merged)):
        date = df_merged.index[i]
        close_price = df_merged['Close'].iloc[i]
        signal = df_merged['Signal'].iloc[i]

        current_portfolio_value = capital + (btc_holdings * close_price)
        portfolio_value.append(current_portfolio_value)

        # 손절매 로직 (매수 포지션 보유 시)
        if btc_holdings > 0 and close_price < entry_price * (1 - stop_loss_pct):
            if capital == 0: # BTC를 들고있다는 의미
                print(f"{date} 손절매 발동: 현재가 {close_price:.2f} < 진입가 {entry_price:.2f}의 {(1-stop_loss_pct)*100:.0f}%")
                capital += btc_holdings * close_price
                trade_log.append({'Date': date, 'Action': 'STOP_LOSS', 'Price': close_price, 'BTC_Sold': btc_holdings, 'Capital_Left': capital})
                btc_holdings = 0
                entry_price = 0 # 포지션 종료
                signal = "HOLD" # 손절매 후에는 다른 거래 신호 무시 (선택적)

        if signal == "BUY":
            if capital > 0: # 현금 보유 시에만 매수
                amount_to_buy = capital / close_price
                btc_holdings += amount_to_buy
                entry_price = close_price # 진입 가격 기록
                capital = 0
                trade_log.append({'Date': date, 'Action': 'BUY', 'Price': close_price, 'BTC_Bought': amount_to_buy, 'Capital_Left': capital})
        elif signal == "SELL":
            if btc_holdings > 0: # BTC 보유 시에만 매도
                capital += btc_holdings * close_price
                trade_log.append({'Date': date, 'Action': 'SELL', 'Price': close_price, 'BTC_Sold': btc_holdings, 'Capital_Left': capital})
                btc_holdings = 0
                entry_price = 0 # 포지션 종료
        # HOLD: 아무것도 하지 않음

    # 백테스트 종료 시점에 BTC 보유하고 있으면 현재가로 정산
    if btc_holdings > 0:
        capital += btc_holdings * df_merged['Close'].iloc[-1]
        trade_log.append({'Date': df_merged.index[-1], 'Action': ' 청산', 'Price': df_merged['Close'].iloc[-1], 'BTC_Sold': btc_holdings, 'Capital_Left': capital})
        btc_holdings = 0
        
    final_portfolio_value = capital # 이미 위에서 계산됨
    total_return = (final_portfolio_value - initial_capital) / initial_capital * 100

    print(f"백테스팅 완료. 최종 포트폴리오 가치: ${final_portfolio_value:.2f}")
    print(f"총 수익률: {total_return:.2f}%")

    # 비교를 위한 바이앤홀드(Buy and Hold) 수익률 계산
    buy_and_hold_return = (df_merged['Close'].iloc[-1] / df_merged['Close'].iloc[0] - 1) * 100
    print(f"바이앤홀드 수익률: {buy_and_hold_return:.2f}%")

    # 기본 성능 지표 (전체 위험 지표는 간소화를 위해 생략)
    # 최대 낙폭 (Maximum Drawdown, MDD) 계산 [1]
    if not portfolio_value: # 포트폴리오 가치 기록이 없으면 MDD 계산 불가
        max_drawdown = 0.0
        print("포트폴리오 가치 기록이 없어 MDD를 계산할 수 없습니다.")
    else:
        portfolio_series = pd.Series(portfolio_value, index=df_merged.index[:len(portfolio_value)])
        rolling_max = portfolio_series.expanding(min_periods=1).max()
        daily_drawdown = (portfolio_series / rolling_max) - 1.0
        max_drawdown = daily_drawdown.min() * 100

    print(f"최대 낙폭: {max_drawdown:.2f}%")

    return {
        'final_value': final_portfolio_value,
        'total_return_percent': total_return,
        'buy_and_hold_return_percent': buy_and_hold_return,
        'max_drawdown_percent': max_drawdown,
        'trade_log': pd.DataFrame(trade_log),
        'portfolio_value_history': portfolio_series if portfolio_value else pd.Series()
    }

# --- 메인 실행 ---
def main():
    # 날짜 범위 정의
    start_date = '2018-01-01'
    # end_date = '2024-01-01' # 원본
    end_date = '2021-01-01' # 테스트를 위해 데이터 기간 단축 (MDD 확인 용이)
    sequence_length = 10 # LSTM/GRU의 시퀀스 길이 [1]
    epochs_per_train = 5 # 빠른 실행을 위해 감소, 보고서는 최대 200 [1]
    batch_size = 32 # 보고서는 16-64 [1]
    learning_rate = 0.001 # 보고서는 0.001-0.01 [1]
    stop_loss_percentage = 0.05 # 손절매 비율 (예: 5%)

    print("=== AI 기반 앙상블 비트코인 트레이딩 전략 시작 ===")
    
    # 1. 데이터 수집
    print("1단계: 데이터 수집")
    df_btc = get_bitcoin_data(start_date, end_date)
    df_trends = get_google_trends_data('bitcoin', start_date, end_date)
    print(f"비트코인 데이터: {df_btc.shape}, Google 트렌드 데이터: {df_trends.shape}")

    # 데이터 병합 및 전처리
    print("2단계: 데이터 전처리")
    X_scaled_df, y, scaler, original_df = preprocess_data(df_btc, df_trends)
    print(f"전처리 후 데이터: X={X_scaled_df.shape}, y={y.shape}")

    # 랜덤 포레스트 예측을 특징으로 추가
    print("3단계: 랜덤 포레스트 훈련")
    rf_predictor = RandomForestPredictor()
    # 이 예시에서는 간단화를 위해 RF를 전체 데이터셋에 한 번 훈련합니다.
    # 더 견고한 접근 방식은 각 롤링 윈도우 내에서 RF를 훈련하는 것입니다.
    rf_predictor.train(X_scaled_df, y)
    rf_predictions_proba = rf_predictor.predict_proba(X_scaled_df)
    print(f"랜덤 포레스트 예측 완료: {len(rf_predictions_proba)} 개 예측")
    
    # 랜덤 포레스트 예측을 X_scaled_df에 특징으로 추가하여 NN 훈련에 사용 [1]
    X_scaled_df['RF_Prediction'] = rf_predictions_proba

    # 모든 특징 목록 업데이트 (RF 예측 포함)
    all_features = X_scaled_df.columns.tolist()
    print(f"모든 특징: {all_features}")

    # 모델 초기화
    print("4단계: 신경망 모델 초기화")
    input_dim_fnn = len(all_features)
    input_dim_rnn = len(all_features) # LSTM/GRU도 모든 특징을 시퀀스 입력으로 받습니다.

    fnn_model = FNN(input_dim_fnn)
    lstm_model = LSTMModel(input_dim_rnn)
    gru_model = GRUModel(input_dim_rnn)
    ensemble_model = EnsembleModel(fnn_model, lstm_model, gru_model)
    print("모델 초기화 완료")

    criterion = nn.BCELoss() # 이진 분류를 위한 이진 교차 엔트로피 손실
    fnn_optimizer = optim.Adam(fnn_model.parameters(), lr=learning_rate)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate) # LSTM에 Adam 사용 [1]
    gru_optimizer = optim.Adam(gru_model.parameters(), lr=learning_rate) # GRU에 Adam 사용 (가정)

    # 시퀀스 생성을 위해 데이터 준비
    print("5단계: 시퀀스 데이터 준비")
    X_seq_all, y_seq_all = create_sequences(X_scaled_df, y, sequence_length)
    print(f"시퀀스 생성 완료: {X_seq_all.shape}, {y_seq_all.shape}")

    # PyTorch 텐서로 변환
    X_seq_all_tensor = torch.tensor(X_seq_all, dtype=torch.float32)
    y_seq_all_tensor = torch.tensor(y_seq_all, dtype=torch.float32)
    # FNN 입력은 시퀀스가 아니므로, 시퀀스 길이만큼 오프셋된 X_scaled_df를 사용합니다.
    X_fnn_all_tensor = torch.tensor(X_scaled_df.iloc[sequence_length:].values, dtype=torch.float32)
    print("텐서 변환 완료")

    # 간소화된 훈련/테스트 분할 (데모 목적)
    # 보고서에 언급된 "롤링 윈도우 방법론"과 "5겹 교차 검증"은 더 복잡하고 계산 비용이 많이 듭니다. [1]
    # 실제 구현에서는 TimeSeriesSplit을 사용하여 훈련 윈도우를 동적으로 슬라이딩하고 재훈련해야 합니다. [4, 5]
    train_size = int(len(X_seq_all_tensor) * 0.8)
    X_fnn_train, X_fnn_test = X_fnn_all_tensor[:train_size], X_fnn_all_tensor[train_size:]
    X_seq_train, X_seq_test = X_seq_all_tensor[:train_size], X_seq_all_tensor[train_size:]
    y_train, y_test = y_seq_all_tensor[:train_size], y_seq_all_tensor[train_size:]

    # DataLoader 생성
    fnn_train_dataset = TensorDataset(X_fnn_train, y_train)
    fnn_train_loader = DataLoader(fnn_train_dataset, batch_size=batch_size, shuffle=True)

    rnn_train_dataset = TensorDataset(X_seq_train, y_train)
    rnn_train_loader = DataLoader(rnn_train_dataset, batch_size=batch_size, shuffle=True)

    fnn_test_dataset = TensorDataset(X_fnn_test, y_test)
    fnn_test_loader = DataLoader(fnn_test_dataset, batch_size=batch_size, shuffle=False)

    rnn_test_dataset = TensorDataset(X_seq_test, y_test)
    rnn_test_loader = DataLoader(rnn_test_dataset, batch_size=batch_size, shuffle=False)

    # 개별 모델 훈련
    train_pytorch_model(fnn_model, fnn_train_loader, criterion, fnn_optimizer, epochs=epochs_per_train, model_name="FNN")
    train_pytorch_model(lstm_model, rnn_train_loader, criterion, lstm_optimizer, epochs=epochs_per_train, model_name="LSTM")
    train_pytorch_model(gru_model, rnn_train_loader, criterion, gru_optimizer, epochs=epochs_per_train, model_name="GRU")

    # 테스트 세트에서 개별 모델의 예측 가져오기
    print("6단계: 개별 모델 예측")
    fnn_preds = evaluate_pytorch_model(fnn_model, fnn_test_loader, model_name="FNN")
    lstm_preds = evaluate_pytorch_model(lstm_model, rnn_test_loader, model_name="LSTM")
    gru_preds = evaluate_pytorch_model(gru_model, rnn_test_loader, model_name="GRU")
    
    print(f"FNN 예측 형태: {np.array(fnn_preds).shape}")
    print(f"LSTM 예측 형태: {np.array(lstm_preds).shape}")
    print(f"GRU 예측 형태: {np.array(gru_preds).shape}")

    # 앙상블 예측 결합 (테스트 세트 평가용)
    if len(fnn_preds) > 0 and len(lstm_preds) > 0 and len(gru_preds) > 0:
        ensemble_preds = (0.4 * np.array(fnn_preds) + 0.3 * np.array(lstm_preds) + 0.3 * np.array(gru_preds))
        
        # 앙상블 평가 (이진 분류기로서)
        ensemble_binary_preds = (ensemble_preds >= 0.5).astype(int)
        y_test_np = y_test.cpu().numpy()

        print(f"\n앙상블 예측 형태: {ensemble_preds.shape}")
        print(f"테스트 레이블 형태: {y_test_np.shape}")
        
        if len(ensemble_binary_preds) == len(y_test_np):
            print("\n앙상블 모델 테스트 세트 성능:")
            print(f"정확도: {accuracy_score(y_test_np, ensemble_binary_preds):.4f}")
            print(f"정밀도: {precision_score(y_test_np, ensemble_binary_preds, zero_division=0):.4f}")
            print(f"재현율: {recall_score(y_test_np, ensemble_binary_preds, zero_division=0):.4f}")
            print(f"F1-점수: {f1_score(y_test_np, ensemble_binary_preds, zero_division=0):.4f}")
        else:
            print(f"예측과 레이블 길이 불일치: 예측={len(ensemble_binary_preds)}, 레이블={len(y_test_np)}")
    else:
        print("예측 결과가 비어있습니다.")

    # --- 전체 데이터셋에 대한 신호 생성 (백테스팅용) ---
    print("\n7단계: 백테스팅을 위한 거래 신호 생성")
    # 전체 데이터셋에 대한 개별 모델의 예측 가져오기 (시퀀스 생성 후)
    fnn_full_dataset = TensorDataset(X_fnn_all_tensor, y_seq_all_tensor)
    fnn_full_loader = DataLoader(fnn_full_dataset, batch_size=batch_size, shuffle=False)

    rnn_full_dataset = TensorDataset(X_seq_all_tensor, y_seq_all_tensor)
    rnn_full_loader = DataLoader(rnn_full_dataset, batch_size=batch_size, shuffle=False)

    fnn_full_preds = evaluate_pytorch_model(fnn_model, fnn_full_loader, model_name="FNN")
    lstm_full_preds = evaluate_pytorch_model(lstm_model, rnn_full_loader, model_name="LSTM")
    gru_full_preds = evaluate_pytorch_model(gru_model, rnn_full_loader, model_name="GRU")

    # 최종 앙상블 예측 (가격 상승 확률)
    if len(fnn_full_preds) > 0 and len(lstm_full_preds) > 0 and len(gru_full_preds) > 0:
        ml_ensemble_signal_proba = (0.4 * np.array(fnn_full_preds) + 0.3 * np.array(lstm_full_preds) + 0.3 * np.array(gru_full_preds))
        print(f"앙상블 신호 생성 완료: {len(ml_ensemble_signal_proba)} 개 신호")
        
        # 신호 생성을 위한 데이터 준비
        # original_df는 RSI, MACD_Hist, GoogleTrends를 포함합니다.
        # X_scaled_df는 스케일링된 버전을 포함합니다.
        # generate_trading_signal 함수에 원본 RSI, MACD_Hist 값과 스케일링된 GoogleTrends 값을 전달합니다.
        # ml_ensemble_signal_proba는 앙상블 모델의 예측입니다.

        # 예측 길이와 일치하도록 original_df를 정렬
        df_for_signals = original_df.iloc[sequence_length:].copy()
        
        if len(df_for_signals) == len(ml_ensemble_signal_proba):
            df_for_signals['ML_Ensemble_Proba'] = ml_ensemble_signal_proba

            signals_df = pd.DataFrame(index=df_for_signals.index)
            signals_df['Signal'] = "HOLD"
            signals_df['Score'] = 0.0

            # X_scaled_df에서 관련 기간에 대한 스케일링된 Google Trends 추출
            scaled_google_trends = X_scaled_df.iloc[sequence_length:]['GoogleTrends']

            print("거래 신호 생성 중...")
            signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
            
            for idx, row in df_for_signals.iterrows():
                # 신호 해석을 위해 원본 RSI 및 MACD_Hist 사용
                rsi_val = row['RSI']
                macd_hist_val = row['MACD_Hist']
                google_trends_val_scaled = scaled_google_trends.loc[idx] # 스케일링된 Google Trends 사용
                ml_ensemble_proba = row['ML_Ensemble_Proba']

                signal, score = generate_trading_signal(rsi_val, macd_hist_val, google_trends_val_scaled, ml_ensemble_proba)
                signals_df.loc[idx, 'Signal'] = signal
                signals_df.loc[idx, 'Score'] = score
                signal_counts[signal] += 1
                
                # 처음 5개 신호에 대한 상세 정보 출력
                if len(signals_df) <= 5:
                    print(f"  신호 {len(signals_df)}: RSI={rsi_val:.2f}, MACD={macd_hist_val:.4f}, "
                          f"GT_scaled={google_trends_val_scaled:.3f}, ML_prob={ml_ensemble_proba:.3f} "
                          f"=> {signal} (점수: {score:.3f})")
            
            print(f"신호 생성 완료: BUY={signal_counts['BUY']}, SELL={signal_counts['SELL']}, HOLD={signal_counts['HOLD']}")
            
            # 백테스팅 전에 거래 신호 요약 출력
            if signal_counts['BUY'] == 0 and signal_counts['SELL'] == 0:
                print("⚠️ 경고: 매수/매도 신호가 전혀 생성되지 않았습니다. 모든 신호가 HOLD입니다.")
                print("이는 다음 중 하나의 원인일 수 있습니다:")
                print("1. 신호 생성 임계값이 너무 높음 (현재: weighted_score > 0.5 또는 < -0.5)")
                print("2. 입력 데이터의 범위가 예상과 다름")
                print("3. 모델 예측 성능이 낮음")
                
                # 실제 weighted_score 분포 확인
                print(f"\n실제 생성된 점수 범위: {signals_df['Score'].min():.3f} ~ {signals_df['Score'].max():.3f}")
                print(f"점수 평균: {signals_df['Score'].mean():.3f}")
                
                # 임계값을 낮춰서 다시 시도
                print("\n임계값을 낮춰서 재시도합니다...")
                for idx, row in df_for_signals.iterrows():
                    score = signals_df.loc[idx, 'Score']
                    # 더 낮은 임계값 사용
                    if score > 0.1:
                        signals_df.loc[idx, 'Signal'] = "BUY"
                    elif score < -0.1:
                        signals_df.loc[idx, 'Signal'] = "SELL"
                    # HOLD는 그대로 유지
                
                # 새로운 신호 카운트
                new_signal_counts = signals_df['Signal'].value_counts().to_dict()
                print(f"낮은 임계값으로 재생성된 신호: {new_signal_counts}")
            
            # 8. 백테스팅
            print("8단계: 백테스팅 실행")
            backtest_results = backtest_strategy(original_df, signals_df, stop_loss_pct=stop_loss_percentage)

            print("\n=== 백테스트 결과 요약 ===")
            print(f"최종 포트폴리오 가치: ${backtest_results['final_value']:.2f}")
            print(f"총 수익률: {backtest_results['total_return_percent']:.2f}%")
            print(f"바이앤홀드 수익률: {backtest_results['buy_and_hold_return_percent']:.2f}%")
            print(f"최대 낙폭: {backtest_results['max_drawdown_percent']:.2f}%")
            print("=== 분석 완료 ===")
        else:
            print(f"데이터 길이 불일치: 신호 데이터={len(df_for_signals)}, 예측={len(ml_ensemble_signal_proba)}")
    else:
        print("전체 데이터셋 예측 실패")

if __name__ == "__main__":
    main()
