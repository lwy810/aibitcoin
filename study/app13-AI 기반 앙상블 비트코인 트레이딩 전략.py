import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os # CSV 파일 존재 여부 확인을 위해 추가

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- 2. 특징 공학 및 전처리 (CSV 로딩 시 이 함수의 역할이 변경됨) ---
def preprocess_data_from_df(input_df, use_google_trends_feature=True):
    """
    이미 대부분 전처리된 DataFrame을 받아 특징 선택, 스케일링을 수행합니다.
    input_df는 'app13-getCsv.py'를 통해 생성된 CSV에서 로드된 데이터라고 가정합니다.
    """
    print("데이터프레임으로부터 전처리 시작 (스케일링 등)...")
    df = input_df.copy()

    # NextDayPriceIncrease 컬럼이 있는지 확인하고 y로 사용
    if 'NextDayPriceIncrease' not in df.columns:
        raise ValueError("입력 DataFrame에 'NextDayPriceIncrease' 컬럼이 없습니다.")
    y = df['NextDayPriceIncrease']

    # 모델을 위한 특징 선택
    features = ['Price_Change_Pct', 'Volume', 'RSI', 'MACD_Hist']
    if use_google_trends_feature:
        if 'GoogleTrends' in df.columns and not df['GoogleTrends'].isnull().all():
            if not df['GoogleTrends'].fillna(0).eq(0).all():
                features.append('GoogleTrends')
                print("GoogleTrends를 학습 특징으로 포함합니다.")
            else:
                print("GoogleTrends 데이터가 모두 0 또는 NaN이므로 학습 특징에 포함하지 않습니다.")
        else:
            print("GoogleTrends 컬럼이 없거나 모두 NaN이므로 학습 특징에 포함하지 않습니다.")
    
    # 선택된 특징들이 df에 모두 존재하는지 확인
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"입력 DataFrame에 다음 특징들이 없습니다: {missing_features}")
        
    X = df[features]
    
    # 결측치 한 번 더 확인 (매우 중요)
    if X.isnull().values.any():
        print("경고: 특징 데이터(X)에 NaN 값이 있습니다. ffill/bfill을 다시 적용합니다.")
        X = X.ffill().bfill()
        if X.isnull().values.any():
             raise ValueError("ffill/bfill 후에도 특징 데이터(X)에 NaN 값이 남아있습니다. CSV 데이터 확인 필요.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)

    # 원본 df (original_df_processed) 준비 - 신호 생성 시 필요
    # 이 df는 스케일링되지 않은 'Close', 'RSI', 'MACD_Hist' 등을 포함해야 함.
    # 또한, 스케일링된 Google Trends 값 ('GoogleTrends_Scaled')도 필요.
    original_df_processed_for_signal = df.copy() # X_scaled_df와 동일한 인덱스를 가짐

    if 'GoogleTrends' in df.columns and not df['GoogleTrends'].isnull().all():
        if 'GoogleTrends_Scaled' not in original_df_processed_for_signal.columns or original_df_processed_for_signal['GoogleTrends_Scaled'].isnull().all():
            print("'GoogleTrends_Scaled' 컬럼을 생성합니다 (원본 GoogleTrends 스케일링)...")
            gt_scaler = StandardScaler()
            # NaN이 아닌 GoogleTrends 값만 스케일링
            valid_gt_idx = df['GoogleTrends'].notna()
            if valid_gt_idx.any():
                 original_df_processed_for_signal.loc[valid_gt_idx, 'GoogleTrends_Scaled'] = gt_scaler.fit_transform(df.loc[valid_gt_idx, ['GoogleTrends']])
                 original_df_processed_for_signal['GoogleTrends_Scaled'] = original_df_processed_for_signal['GoogleTrends_Scaled'].fillna(0) # 스케일링 후 NaN은 0으로
            else:
                original_df_processed_for_signal['GoogleTrends_Scaled'] = 0
        else:
            print("기존 'GoogleTrends_Scaled' 컬럼을 사용합니다.")
            original_df_processed_for_signal['GoogleTrends_Scaled'] = original_df_processed_for_signal['GoogleTrends_Scaled'].fillna(0) # 혹시 모를 NaN 처리
    else:
        original_df_processed_for_signal['GoogleTrends_Scaled'] = 0
        
    print(f"데이터프레임으로부터 전처리 완료. X 형태: {X_scaled_df.shape}, y 형태: {y.shape}")
    return X_scaled_df, y, scaler, original_df_processed_for_signal

def create_sequences(data, target, sequence_length=10):
    """
    롤링 윈도우를 기반으로 LSTM/GRU 모델을 위한 시퀀스를 생성합니다.
    data: 스케일링된 특징 DataFrame (예: X_scaled_df_nn)
    target: 예측 대상 Series (예: y)
    """
    xs, ys = [], []
    # data와 target의 인덱스가 일치한다고 가정하고, 길이를 기준으로 루프
    # target은 data보다 sequence_length 만큼 짧게 사용될 수 있음 (미래 값을 예측하므로)
    # 또는, data에서 target을 만들 때 이미 처리되었을 수 있음.
    # 여기서는 data (X_scaled_df_nn)와 target (y)이 create_sequences 호출 전에 
    # 동일한 시작점을 가지도록 정렬/조정되었다고 가정함.
    for i in range(len(data) - sequence_length):
        x = data.iloc[i:(i + sequence_length)].values
        # y_val은 target Series에서 직접 인덱싱 (target이 이미 올바른 시점의 값을 가진다고 가정)
        y_val = target.iloc[i + sequence_length] 
        xs.append(x)
        ys.append(y_val)
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
def generate_trading_signal(rsi_val, macd_hist_val, google_trends_scaled_val, rf_signal_proba, use_google_trends_in_signal=True):
    """
    가중치 점수를 계산하고 매수/매도/보유 신호를 결정합니다.
    RF 예측 확률을 기반으로 ML 신호를 생성합니다.
    """
    # RSI 신호
    if rsi_val < 30:
        rsi_signal = 1.0
    elif rsi_val > 70:
        rsi_signal = -1.0
    else:
        rsi_signal = 1 - 2 * ((rsi_val - 30) / 40)
        rsi_signal = np.clip(rsi_signal, -1.0, 1.0)

    # MACD 신호
    if macd_hist_val > 0:
        macd_signal = 1.0
    elif macd_hist_val < 0:
        macd_signal = -1.0
    else:
        macd_signal = 0.0

    # Google 트렌드 신호 (논문 방식 적용)
    google_trends_signal = 0.0
    if use_google_trends_in_signal and google_trends_scaled_val != 0: # 0이 아니면 유효한 값으로 간주
        # 스케일링된 값이 0보다 크면 +1, 작으면 -1 (논문에서는 7일 이평선 초과/미만)
        # 여기서는 스케일링된 값의 부호로 단순화
        google_trends_signal = 1.0 if google_trends_scaled_val > 0 else -1.0

    # ML 신호 (Random Forest 예측 확률 기반)
    # rf_signal_proba는 가격 상승 확률이므로, 0.5를 기준으로 신호 결정
    ml_final_signal = 1.0 if rf_signal_proba > 0.5 else -1.0

    # 최종 신호 가중치 (논문 및 이전 논의 기반)
    # 기존: TA: 0.4, Google Trends: 0.2, ML: 0.4
    # 여기서는 TA_Combined_Signal, GoogleTrends_Signal, ML_Signal (RF)
    
    ta_combined_signal = (rsi_signal * 0.5) + (macd_signal * 0.5) # RSI와 MACD는 동일 가중치

    # 가중치: TA 40%, Google Trends 20%, ML(RF) 40%
    weights = {'ta': 0.4, 'google_trends': 0.2, 'ml': 0.4}
    
    if not use_google_trends_in_signal:
        # Google Trends를 사용하지 않으면 해당 가중치를 TA와 ML에 분배
        weights['ta'] += weights['google_trends'] * 0.5 # TA에 절반
        weights['ml'] += weights['google_trends'] * 0.5 # ML에 절반
        weights['google_trends'] = 0
        google_trends_signal = 0 # 명시적으로 0으로 설정

    final_weighted_signal = (ta_combined_signal * weights['ta'] +
                             google_trends_signal * weights['google_trends'] +
                             ml_final_signal * weights['ml'])
    
    # 최종 결정: 매수, 매도, 보유
    if final_weighted_signal > 0.3: # 매수 임계값 (튜닝 가능)
        return 1
    elif final_weighted_signal < -0.3: # 매도 임계값 (튜닝 가능)
        return -1
    else:
        return 0

# --- 8. 백테스팅 ---
def backtest_strategy(df_original_for_backtest, signals_df, initial_capital=10000, stop_loss_pct=0.03, transaction_cost_pct=0.005):
    """
    거래 전략의 간소화된 백테스트를 수행합니다.
    df_original_for_backtest: 'Close' 가격 등 원본 데이터가 포함된 DataFrame.
    """
    print(f"백테스팅 시작 (손절매: {stop_loss_pct*100}%, 거래비용: {transaction_cost_pct*100}%)")
    capital = initial_capital
    btc_holdings = 0
    entry_price = 0 
    portfolio_value = []
    trade_log = []

    # signals_df의 인덱스를 사용하여 df_original_for_backtest에서 해당 기간의 데이터만 사용
    # df_merged는 이제 signals_df와 동일한 인덱스를 가지며 'Close' 가격을 포함해야 함
    if not signals_df.index.isin(df_original_for_backtest.index).all():
        raise ValueError("signals_df의 일부 인덱스가 df_original_for_backtest에 없습니다.")
    
    df_merged = df_original_for_backtest.loc[signals_df.index].copy() # 해당 기간의 원본 데이터 사용
    df_merged['Signal'] = signals_df['Signal']
    df_merged['Score'] = signals_df['Score']

    for i in range(len(df_merged)):
        date = df_merged.index[i]
        close_price = df_merged['Close'].iloc[i]
        signal = df_merged['Signal'].iloc[i]

        current_portfolio_value = capital + (btc_holdings * close_price)
        portfolio_value.append(current_portfolio_value)

        if btc_holdings > 0 and close_price < entry_price * (1 - stop_loss_pct):
            if capital == 0: 
                print(f"{date} 손절매 발동: 현재가 {close_price:.2f} < 진입가 {entry_price:.2f}의 {(1-stop_loss_pct)*100:.0f}%")
                proceeds = btc_holdings * close_price
                cost = proceeds * transaction_cost_pct
                capital += (proceeds - cost)
                trade_log.append({'Date': date, 'Action': 'STOP_LOSS', 'Price': close_price, 'BTC_Sold': btc_holdings, 'Cost': cost, 'Capital_Left': capital})
                btc_holdings = 0
                entry_price = 0 
                signal = "HOLD" 

        if signal == "BUY":
            if capital > 0: 
                cost_of_purchase = capital * transaction_cost_pct 
                available_capital_for_btc = capital - cost_of_purchase
                if available_capital_for_btc > 0 : 
                    amount_to_buy = available_capital_for_btc / close_price
                    btc_holdings += amount_to_buy
                    entry_price = close_price 
                    capital = 0 
                    trade_log.append({'Date': date, 'Action': 'BUY', 'Price': close_price, 'BTC_Bought': amount_to_buy, 'Cost': cost_of_purchase, 'Capital_Left': capital})
        elif signal == "SELL":
            if btc_holdings > 0: 
                proceeds = btc_holdings * close_price
                cost = proceeds * transaction_cost_pct
                capital += (proceeds - cost)
                trade_log.append({'Date': date, 'Action': 'SELL', 'Price': close_price, 'BTC_Sold': btc_holdings, 'Cost': cost, 'Capital_Left': capital})
                btc_holdings = 0
                entry_price = 0 

    if btc_holdings > 0:
        final_proceeds = btc_holdings * df_merged['Close'].iloc[-1]
        final_cost = final_proceeds * transaction_cost_pct
        capital += (final_proceeds - final_cost)
        trade_log.append({'Date': df_merged.index[-1], 'Action': ' 청산', 'Price': df_merged['Close'].iloc[-1], 'BTC_Sold': btc_holdings, 'Cost': final_cost, 'Capital_Left': capital})
        btc_holdings = 0
        
    final_portfolio_value = capital
    total_return = (final_portfolio_value - initial_capital) / initial_capital * 100

    print(f"백테스팅 완료. 최종 포트폴리오 가치: ${final_portfolio_value:.2f}")
    print(f"총 수익률: {total_return:.2f}%")

    buy_and_hold_start_price = df_merged['Close'].iloc[0]
    buy_and_hold_end_price = df_merged['Close'].iloc[-1]
    buy_and_hold_capital = initial_capital
    cost_at_buy = buy_and_hold_capital * transaction_cost_pct
    buy_and_hold_btc = (buy_and_hold_capital - cost_at_buy) / buy_and_hold_start_price
    final_value_bh = buy_and_hold_btc * buy_and_hold_end_price
    cost_at_sell_bh = final_value_bh * transaction_cost_pct
    final_value_bh -= cost_at_sell_bh
    buy_and_hold_return = (final_value_bh - initial_capital) / initial_capital * 100
    print(f"바이앤홀드 수익률 (거래비용 고려): {buy_and_hold_return:.2f}%")

    if not portfolio_value: 
        max_drawdown = 0.0
        print("포트폴리오 가치 기록이 없어 MDD를 계산할 수 없습니다.")
        portfolio_series = pd.Series()
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
        'portfolio_value_history': portfolio_series
    }

# --- 메인 실행 ---
def main():
    start_date = '2018-01-01' # CSV 생성 시 사용된 기간과 일치해야 함
    end_date = '2024-01-01'   # CSV 생성 시 사용된 기간과 일치해야 함
    sequence_length = 10
    epochs_per_train = 5 
    batch_size = 32
    learning_rate = 0.001
    stop_loss_percentage = 0.03
    # google_trends_timeframe_months, google_trends_delay_seconds 는 CSV 생성 시 사용되므로 여기선 불필요
    transaction_cost_percentage = 0.005 
    
    use_google_trends_in_model_features = True 
    use_google_trends_in_signal_generation = True 

    csv_filename = 'bitcoin_processed_data.csv'
    # force_regenerate_data_from_api 플래그는 API 호출 로직이 삭제되므로 의미 없어짐

    print(f"=== AI 기반 앙상블 비트코인 트레이딩 전략 시작 (CSV 사용) ===")
    print(f"모델 특징에 GoogleTrends 사용: {use_google_trends_in_model_features}")
    print(f"신호 생성에 GoogleTrends 사용: {use_google_trends_in_signal_generation}")
    print(f"거래비용: {transaction_cost_percentage*100}%")

    processed_df = None
    if not os.path.exists(csv_filename):
        print(f"오류: '{csv_filename}' 파일이 존재하지 않습니다.")
        print(f"먼저 'study/app13-getCsv.py' 스크립트를 실행하여 데이터 파일을 생성해주세요.")
        return # 프로그램 종료
    
    print(f"'{csv_filename}' 파일에서 데이터 로드 중...")
    try:
        processed_df = pd.read_csv(csv_filename, index_col='Date', parse_dates=True)
        print(f"CSV에서 데이터 로드 완료. 형태: {processed_df.shape}")
        
        required_cols = ['Close', 'Volume', 'RSI', 'MACD_Hist', 'Price_Change_Pct', 'NextDayPriceIncrease']
        if use_google_trends_in_model_features or use_google_trends_in_signal_generation:
            if 'GoogleTrends' not in processed_df.columns:
                print(f"경고: Google Trends를 사용하도록 설정되었으나, CSV 파일에 'GoogleTrends' 컬럼이 없습니다.")
                # 필요시 여기서 use_google_trends_in_model_features 와 use_google_trends_in_signal_generation 을 False로 강제 변경 가능
                # 또는, CSV가 잘못되었다고 판단하고 종료할 수도 있음
        
        missing_csv_cols = [col for col in required_cols if col not in processed_df.columns and col != 'GoogleTrends'] # GoogleTrends는 위에서 별도 체크
        if 'GoogleTrends' in required_cols and 'GoogleTrends' not in processed_df.columns:
             pass # GoogleTrends 컬럼이 필수는 아니지만, 사용 설정 시 없으면 위에서 경고 impres
        elif missing_csv_cols: # GoogleTrends를 제외한 필수 컬럼이 없는 경우
            print(f"오류: CSV 파일에 필요한 컬럼이 부족합니다: {missing_csv_cols}.")
            print(f"'{csv_filename}' 파일을 확인하거나 'study/app13-getCsv.py'를 다시 실행해주세요.")
            return # 프로그램 종료
            
    except Exception as e:
        print(f"CSV 파일 ('{csv_filename}') 로드 중 오류: {e}")
        print(f"파일이 손상되었거나, 형식이 잘못되었을 수 있습니다. 'study/app13-getCsv.py'를 다시 실행하여 파일을 재생성해보세요.")
        return # 프로그램 종료
    
    if processed_df is None or processed_df.empty: # 혹시 모를 상황 대비
        print(f"데이터 로드에 실패하여 프로그램을 종료합니다. '{csv_filename}' 파일을 확인해주세요.")
        return

    print("2단계: 데이터 전처리 (스케일링 등)")
    X_scaled_df, y, scaler, original_df_processed = preprocess_data_from_df(processed_df, use_google_trends_feature=use_google_trends_in_model_features)
    print(f"전처리 후 데이터: X={X_scaled_df.shape}, y={y.shape}, original_df_processed={original_df_processed.shape}")

    # --- 이하 모델 학습 및 백테스팅 로직은 이전과 거의 동일 ---
    # 다만, original_df_processed를 backtest_strategy에 전달하는 부분 확인 필요
    print("3단계: 랜덤 포레스트 훈련")
    rf_predictor = RandomForestPredictor()
    rf_predictor.train(X_scaled_df, y) 
    rf_predictions_proba = rf_predictor.predict_proba(X_scaled_df)
    print(f"랜덤 포레스트 예측 완료: {len(rf_predictions_proba)} 개 예측")

    if hasattr(rf_predictor.model, 'feature_importances_'):
        importances = rf_predictor.model.feature_importances_
        feature_names = X_scaled_df.columns 
        sorted_indices = np.argsort(importances)[::-1]
        print("\nRandomForest 특징 중요도:")
        for i in sorted_indices:
            print(f"{feature_names[i]}: {importances[i]:.4f}")
        print("-"*30)
    
    X_scaled_df_nn = X_scaled_df.copy()
    X_scaled_df_nn['RF_Prediction'] = rf_predictions_proba

    all_features_nn = X_scaled_df_nn.columns.tolist()
    print(f"모든 특징 (NN 입력용): {all_features_nn}")

    print("4단계: 신경망 모델 초기화")
    input_dim_fnn = len(all_features_nn)
    input_dim_rnn = len(all_features_nn)

    fnn_model = FNN(input_dim_fnn)
    lstm_model = LSTMModel(input_dim_rnn)
    gru_model = GRUModel(input_dim_rnn)
    print("모델 초기화 완료")

    criterion = nn.BCELoss()
    fnn_optimizer = optim.Adam(fnn_model.parameters(), lr=learning_rate)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
    gru_optimizer = optim.Adam(gru_model.parameters(), lr=learning_rate)

    print("5단계: 시퀀스 데이터 준비")
    # y는 스케일링된 X와 동일한 인덱스를 가져야 함. preprocess_data_from_df에서 처리됨.
    X_seq_all, y_seq_all = create_sequences(X_scaled_df_nn, y, sequence_length)
    print(f"시퀀스 생성 완료: {X_seq_all.shape}, {y_seq_all.shape}")

    X_seq_all_tensor = torch.tensor(X_seq_all, dtype=torch.float32)
    y_seq_all_tensor = torch.tensor(y_seq_all, dtype=torch.float32)
    # X_scaled_df_nn에서 sequence_length 이후의 데이터를 FNN 입력으로 사용
    X_fnn_all_tensor = torch.tensor(X_scaled_df_nn.iloc[sequence_length:].values, dtype=torch.float32)
    print("텐서 변환 완료")
    
    train_size = int(len(X_seq_all_tensor) * 0.8)
    X_fnn_train, X_fnn_test = X_fnn_all_tensor[:train_size], X_fnn_all_tensor[train_size:]
    X_seq_train, X_seq_test = X_seq_all_tensor[:train_size], X_seq_all_tensor[train_size:]
    y_train, y_test = y_seq_all_tensor[:train_size], y_seq_all_tensor[train_size:] # y_seq_all_tensor에서 분할

    fnn_train_dataset = TensorDataset(X_fnn_train, y_train)
    fnn_train_loader = DataLoader(fnn_train_dataset, batch_size=batch_size, shuffle=True)

    rnn_train_dataset = TensorDataset(X_seq_train, y_train)
    rnn_train_loader = DataLoader(rnn_train_dataset, batch_size=batch_size, shuffle=True)

    fnn_test_dataset = TensorDataset(X_fnn_test, y_test)
    fnn_test_loader = DataLoader(fnn_test_dataset, batch_size=batch_size, shuffle=False)

    rnn_test_dataset = TensorDataset(X_seq_test, y_test)
    rnn_test_loader = DataLoader(rnn_test_dataset, batch_size=batch_size, shuffle=False)

    train_pytorch_model(fnn_model, fnn_train_loader, criterion, fnn_optimizer, epochs=epochs_per_train, model_name="FNN")
    train_pytorch_model(lstm_model, rnn_train_loader, criterion, lstm_optimizer, epochs=epochs_per_train, model_name="LSTM")
    train_pytorch_model(gru_model, rnn_train_loader, criterion, gru_optimizer, epochs=epochs_per_train, model_name="GRU")

    print("6단계: 개별 모델 예측")
    fnn_preds = evaluate_pytorch_model(fnn_model, fnn_test_loader, model_name="FNN")
    lstm_preds = evaluate_pytorch_model(lstm_model, rnn_test_loader, model_name="LSTM")
    gru_preds = evaluate_pytorch_model(gru_model, rnn_test_loader, model_name="GRU")
    
    print(f"FNN 예측 형태: {np.array(fnn_preds).shape}")
    print(f"LSTM 예측 형태: {np.array(lstm_preds).shape}")
    print(f"GRU 예측 형태: {np.array(gru_preds).shape}")

    if len(fnn_preds) > 0 and len(lstm_preds) > 0 and len(gru_preds) > 0:
        ensemble_preds = (0.4 * np.array(fnn_preds) + 0.3 * np.array(lstm_preds) + 0.3 * np.array(gru_preds))
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
        print("테스트 세트 예측 결과가 비어있습니다.")

    print(f"\n7단계: 백테스팅을 위한 거래 신호 생성 (Google Trends 신호 생성 시 사용: {use_google_trends_in_signal_generation})")
    # 전체 데이터에 대한 예측 (FNN, LSTM, GRU)
    # y_seq_all_tensor는 X_seq_all_tensor에 대한 레이블
    fnn_full_dataset = TensorDataset(X_fnn_all_tensor, y_seq_all_tensor) 
    fnn_full_loader = DataLoader(fnn_full_dataset, batch_size=batch_size, shuffle=False)
    rnn_full_dataset = TensorDataset(X_seq_all_tensor, y_seq_all_tensor)
    rnn_full_loader = DataLoader(rnn_full_dataset, batch_size=batch_size, shuffle=False)

    fnn_full_preds = evaluate_pytorch_model(fnn_model, fnn_full_loader, model_name="FNN")
    lstm_full_preds = evaluate_pytorch_model(lstm_model, rnn_full_loader, model_name="LSTM")
    gru_full_preds = evaluate_pytorch_model(gru_model, rnn_full_loader, model_name="GRU")

    if len(fnn_full_preds) > 0 and len(lstm_full_preds) > 0 and len(gru_full_preds) > 0:
        ml_ensemble_signal_proba = (0.4 * np.array(fnn_full_preds) + 0.3 * np.array(lstm_full_preds) + 0.3 * np.array(gru_full_preds))
        print(f"전체 데이터에 대한 앙상블 신호 생성 완료: {len(ml_ensemble_signal_proba)} 개 신호")
        
        # df_for_signals는 original_df_processed에서 시퀀스 길이를 제외한 부분
        # original_df_processed는 preprocess_data_from_df의 반환값으로, X_scaled_df와 동일한 인덱스를 가짐
        # X_fnn_all_tensor와 ml_ensemble_signal_proba는 X_scaled_df_nn.iloc[sequence_length:] 기준으로 생성됨.
        # 따라서 original_df_processed.iloc[sequence_length:]를 사용해야 인덱스가 맞음.
        df_for_signals = original_df_processed.iloc[sequence_length:].copy()

        if len(df_for_signals) == len(ml_ensemble_signal_proba):
            df_for_signals['ML_Ensemble_Proba'] = ml_ensemble_signal_proba
            signals_df = pd.DataFrame(index=df_for_signals.index)
            signals_df['Signal'] = "HOLD"
            signals_df['Score'] = 0.0

            print("거래 신호 생성 중...")
            signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
            
            for idx, row in df_for_signals.iterrows():
                rsi_val = row['RSI']
                macd_hist_val = row['MACD_Hist']
                
                google_trends_val_scaled_for_signal = 0 
                if use_google_trends_in_signal_generation and 'GoogleTrends_Scaled' in row and pd.notna(row['GoogleTrends_Scaled']):
                    google_trends_val_scaled_for_signal = row['GoogleTrends_Scaled']
                elif use_google_trends_in_signal_generation:
                    print(f"경고: {idx} 날짜의 GoogleTrends_Scaled 값이 NaN입니다. 0으로 처리됩니다.")
                
                ml_ensemble_proba_val = row['ML_Ensemble_Proba']

                signal, score = generate_trading_signal(rsi_val, macd_hist_val, 
                                                        google_trends_val_scaled_for_signal, 
                                                        ml_ensemble_proba_val,
                                                        use_google_trends_in_signal=use_google_trends_in_signal_generation)
                signals_df.loc[idx, 'Signal'] = signal
                signals_df.loc[idx, 'Score'] = score
                signal_counts[signal] += 1
            
            print(f"신호 생성 완료: BUY={signal_counts['BUY']}, SELL={signal_counts['SELL']}, HOLD={signal_counts['HOLD']}")
            
            if signal_counts['BUY'] == 0 and signal_counts['SELL'] == 0:
                print("⚠️ 경고: 매수/매도 신호가 전혀 생성되지 않았습니다...") 
                print(f"\n실제 생성된 점수 범위: {signals_df['Score'].min():.3f} ~ {signals_df['Score'].max():.3f}")
                print(f"점수 평균: {signals_df['Score'].mean():.3f}")
                print("\n임계값을 낮춰서 재시도합니다...")
                # 낮은 임계값으로 재시도하는 로직은 일단 유지
                # ...
            
            print("8단계: 백테스팅 실행")
            # backtest_strategy는 원본 가격 데이터가 포함된 original_df_processed를 사용해야 함
            backtest_results = backtest_strategy(original_df_processed, signals_df, 
                                                 initial_capital=10000, 
                                                 stop_loss_pct=stop_loss_percentage,
                                                 transaction_cost_pct=transaction_cost_percentage)

            print("\n=== 백테스트 결과 요약 ===")
            print(f"최종 포트폴리오 가치: ${backtest_results['final_value']:.2f}")
            print(f"총 수익률: {backtest_results['total_return_percent']:.2f}%")
            print(f"바이앤홀드 수익률: {backtest_results['buy_and_hold_return_percent']:.2f}%")
            print(f"최대 낙폭: {backtest_results['max_drawdown_percent']:.2f}%")
            print(f"=== 분석 완료 ===")
        else:
            print(f"데이터 길이 불일치 오류: df_for_signals ({len(df_for_signals)}) vs ml_ensemble_signal_proba ({len(ml_ensemble_signal_proba)})")
            print("original_df_processed.iloc[sequence_length:] 와 X_scaled_df_nn.iloc[sequence_length:] 의 인덱스/길이가 일치하는지 확인 필요.")
    else:
        print("전체 데이터셋에 대한 예측 실패")

    # 최종 ML 신호는 RF 예측 확률을 사용
    ml_signal_probas_for_backtest = rf_predictions_proba

    # --- 6. 거래 신호 생성 및 백테스팅 ---\n
    print("\\\\n--- 최종 거래 신호 생성 및 백테스팅 (RF 모델 기반) ---")\n
    backtest_results = backtest_strategy(original_df_processed, signals_df, 
                                         initial_capital=10000, 
                                         stop_loss_pct=stop_loss_percentage,
                                         transaction_cost_pct=transaction_cost_percentage)

    print("\n=== 백테스트 결과 요약 ===")
    print(f"최종 포트폴리오 가치: ${backtest_results['final_value']:.2f}")
    print(f"총 수익률: {backtest_results['total_return_percent']:.2f}%")
    print(f"바이앤홀드 수익률: {backtest_results['buy_and_hold_return_percent']:.2f}%")
    print(f"최대 낙폭: {backtest_results['max_drawdown_percent']:.2f}%")
    print(f"=== 분석 완료 ===")

if __name__ == "__main__":
    main()
