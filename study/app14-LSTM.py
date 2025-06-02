import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# GPU 사용 설정 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 시퀀스 데이터 생성 함수
def create_sequences(features, target, sequence_length=10):
    """
    LSTM 모델을 위한 시퀀스 데이터를 생성합니다.
    features: 스케일링된 특징 DataFrame
    target: 타겟 Series
    sequence_length: 입력 시퀀스의 길이
    """
    xs, ys = [], []
    for i in range(len(features) - sequence_length):
        x = features.iloc[i:(i + sequence_length)].values
        y = target.iloc[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# LSTM 모델 정의
class LSTMPricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, output_dim=1, dropout_rate=0.2):
        super(LSTMPricePredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid() # 이진 분류를 위한 시그모이드 활성화 함수

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0)) # out: (batch_size, seq_len, hidden_dim)
        out = self.fc(out[:, -1, :])    # 마지막 타임스텝의 출력만 사용: (batch_size, output_dim)
        out = self.sigmoid(out)
        return out

# 모델 훈련 함수
def train_model(model, train_loader, criterion, optimizer, epochs=100, model_name="LSTM"):
    print(f"\n{model_name} 모델 훈련 시작 (총 {epochs} 에포크)...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
    print(f"{model_name} 모델 훈련 완료.")

# 모델 평가 함수
def evaluate_model(model, test_loader, criterion, model_name="LSTM"):
    print(f"\n{model_name} 모델 평가 시작...")
    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target.float().unsqueeze(1))
            test_loss += loss.item()
            
            # 확률값을 0.5 기준으로 이진 분류 (0 또는 1)
            predicted = (output >= 0.5).squeeze().cpu().numpy()
            all_preds.extend(predicted)
            all_targets.extend(target.cpu().numpy())
            
    avg_test_loss = test_loss / len(test_loader)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    print(f"{model_name} 테스트 세트 성능:")
    print(f"  평균 손실: {avg_test_loss:.4f}")
    print(f"  정확도 (Accuracy): {accuracy:.4f}")
    print(f"  정밀도 (Precision): {precision:.4f}")
    print(f"  재현율 (Recall): {recall:.4f}")
    print(f"  F1 점수 (F1 Score): {f1:.4f}")
    return accuracy, precision, recall, f1

def backtest_strategy_lstm(df_price_data, signals_series, initial_capital=10000, 
                           stop_loss_pct=0.08, transaction_cost_pct=0.0005,
                           signal_threshold_buy=0.52, signal_threshold_sell=0.48):
    """
    LSTM 예측 확률을 기반으로 거래 전략의 백테스트를 수행합니다.
    df_price_data: 'Close', 'Open', 'High', 'Low' 가격이 포함된 DataFrame (인덱스는 Date).
    signals_series: LSTM 예측 확률 (0~1)을 담은 Pandas Series (인덱스는 Date).
    initial_capital: 초기 자본금.
    stop_loss_pct: 손절매 비율 (예: 0.08은 8%).
    transaction_cost_pct: 거래 비용 비율 (예: 0.0005는 0.05%).
    signal_threshold_buy: 매수 신호로 간주할 예측 확률 임계값.
    signal_threshold_sell: 매도 신호로 간주할 예측 확률 임계값.
    """
    print(f"\n백테스팅 시작 (최근 데이터 대상)")
    print(f"  초기 자본: ${initial_capital:,.2f}")
    print(f"  손절매: {stop_loss_pct*100:.2f}%")
    print(f"  거래 비용: {transaction_cost_pct*100:.3f}%")
    print(f"  매수 신호 임계값 (확률 >): {signal_threshold_buy}")
    print(f"  매도 신호 임계값 (확률 <): {signal_threshold_sell}")

    capital = initial_capital
    holdings = 0  # 현재 보유 코인 수
    entry_price = 0
    portfolio_values = [] # 일별 포트폴리오 가치 기록
    trade_log = []
    num_wins = 0
    num_losses = 0
    total_profit = 0
    total_loss = 0

    # signals_series와 df_price_data의 인덱스를 기준으로 병합 (날짜 정렬)
    # LSTM 시퀀스 생성으로 인해 signals_series의 시작 날짜가 df_price_data보다 늦을 수 있음
    # 백테스팅은 signals_series가 제공하는 기간에 대해서만 수행
    
    # df_price_data에서 signals_series와 동일한 기간의 데이터만 사용
    # LSTM 예측은 하루 뒤 가격을 예측하므로, 신호 발생일의 종가로 거래한다고 가정
    # signals_series의 인덱스는 예측 대상 날짜의 전날임.
    # 즉, signal_series.index[t]의 신호는 df_price_data.index[t+1] 가격에 대한 예측.
    # 거래는 df_price_data.index[t+1]의 가격으로 이루어짐.

    # 실제 거래가 일어나는 날짜는 signals_series의 날짜보다 하루 뒤
    # signals_series의 길이에 맞춰 df_price_data를 슬라이싱해야 함.
    # LSTM 아웃풋은 y_test와 길이가 같고, y_test는 X_test와 길이가 같음
    # X_test는 원본 데이터에서 sequence_length 이후부터 시작됨.
    # signals_series의 인덱스는 y_test의 인덱스와 동일하게 설정되어야 함.

    # signals_series의 인덱스와 df_price_data의 인덱스가 일치해야 함
    # 일반적으로 LSTM 결과는 (N-seq_len)개의 예측을 생성
    # 이 예측은 (seq_len)번째 날부터 마지막 날까지의 가격 변동에 대한 것임.
    # 따라서 signals_series의 인덱스는 df_price_data의 (seq_len)번째 날부터 시작되어야 함.
    
    # 여기서는 signals_series의 인덱스가 이미 올바르게 설정되었다고 가정하고,
    # df_price_data에서 해당 기간의 데이터를 가져옴.
    valid_price_data = df_price_data.loc[signals_series.index]

    if len(valid_price_data) != len(signals_series):
        print(f"경고: 가격 데이터 길이({len(valid_price_data)})와 신호 시리즈 길이({len(signals_series)}) 불일치. 백테스팅이 정확하지 않을 수 있습니다.")
        # 짧은 쪽에 맞춤 (일반적으로 신호가 더 짧음)
        common_index = signals_series.index.intersection(valid_price_data.index)
        signals_series = signals_series.loc[common_index]
        valid_price_data = valid_price_data.loc[common_index]
        if common_index.empty:
            print("오류: 공통 인덱스가 없어 백테스팅을 진행할 수 없습니다.")
            return {
                'total_return_percent': 0,
                'mdd_percent': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'final_capital': initial_capital,
                'num_trades': 0
            }

    for date, current_price in valid_price_data['Close'].items():
        current_signal_prob = signals_series.loc[date]
        action = 0 # 0: 보유, 1: 매수, -1: 매도

        if current_signal_prob > signal_threshold_buy:
            action = 1 # 매수
        elif current_signal_prob < signal_threshold_sell:
            action = -1 # 매도
        
        # 포트폴리오 가치 계산 (거래 전)
        portfolio_values.append(capital + holdings * current_price)

        # 손절매 로직 (매수 포지션 보유 시)
        if holdings > 0 and current_price < entry_price * (1 - stop_loss_pct):
            sell_amount = holdings
            proceeds = sell_amount * current_price * (1 - transaction_cost_pct)
            capital += proceeds
            
            trade_profit = proceeds - (sell_amount * entry_price * (1 + transaction_cost_pct)) # 단순화된 진입 비용 고려
            if trade_profit > 0: 
                num_wins += 1
                total_profit += trade_profit
            else: 
                num_losses += 1
                total_loss += abs(trade_profit)

            trade_log.append({
                'Date': date, 'Action': 'STOP_LOSS', 'Price': current_price,
                'Amount': sell_amount, 'Capital': capital
            })
            holdings = 0
            entry_price = 0
            action = 0 # 손절매 후에는 당일 추가 행동 없음

        # 거래 실행 - 적응형 투자 전략 적용
        if action == 1 and capital > 0: # 매수
            # 예측 확률에 따라 투자 비율 조정
            if current_signal_prob > 0.65:  # 매우 강한 신호
                investment_ratio = 0.8
            elif current_signal_prob > 0.6:  # 강한 신호
                investment_ratio = 0.6
            else:  # 일반 신호
                investment_ratio = 0.4
                
            invest_amount = capital * investment_ratio
            buy_amount_in_currency = invest_amount / (1 + transaction_cost_pct)
            amount_to_buy = buy_amount_in_currency / current_price
            holdings += amount_to_buy
            capital -= invest_amount  # 투자한 만큼만 차감
            entry_price = current_price
            trade_log.append({
                'Date': date, 'Action': 'BUY', 'Price': current_price,
                'Amount': amount_to_buy, 'Capital': capital, 'Signal_Prob': current_signal_prob
            })
        elif action == -1 and holdings > 0: # 매도
            sell_amount = holdings
            proceeds = sell_amount * current_price * (1 - transaction_cost_pct)
            capital += proceeds
            
            trade_profit = proceeds - (sell_amount * entry_price * (1 + transaction_cost_pct)) # 진입 비용 고려
            if trade_profit > 0: 
                num_wins += 1
                total_profit += trade_profit
            else: 
                num_losses += 1
                total_loss += abs(trade_profit)

            trade_log.append({
                'Date': date, 'Action': 'SELL', 'Price': current_price,
                'Amount': sell_amount, 'Capital': capital
            })
            holdings = 0
            entry_price = 0
    
    # 백테스팅 마지막 날, 보유 코인 있으면 시장가로 청산
    if holdings > 0:
        last_price = valid_price_data['Close'].iloc[-1]
        proceeds = holdings * last_price * (1 - transaction_cost_pct)
        capital += proceeds
        trade_profit = proceeds - (holdings * entry_price * (1 + transaction_cost_pct))
        if trade_profit > 0: 
            num_wins += 1
            total_profit += trade_profit
        else: 
            num_losses += 1
            total_loss += abs(trade_profit)
        
        trade_log.append({
            'Date': valid_price_data.index[-1], 'Action': '최종청산', 'Price': last_price,
            'Amount': holdings, 'Capital': capital
        })
        holdings = 0

    final_capital = capital
    total_return_percent = (final_capital - initial_capital) / initial_capital * 100
    num_trades = len([t for t in trade_log if t['Action'] in ['BUY', 'SELL', 'STOP_LOSS']])

    # MDD 계산 - 인덱스 길이 문제 해결
    if not portfolio_values or len(portfolio_values) == 0:
        mdd_percent = 0.0
        portfolio_series = pd.Series()
    else:
        # portfolio_values와 동일한 길이의 인덱스 생성
        # valid_price_data의 인덱스를 포트폴리오 가치 개수에 맞춰 사용
        portfolio_index = valid_price_data.index[:len(portfolio_values)]
        if len(portfolio_index) != len(portfolio_values):
            print(f"경고: 포트폴리오 가치 개수({len(portfolio_values)})와 인덱스 개수({len(portfolio_index)}) 불일치")
            # 길이를 맞춤 (보수적 접근)
            min_len = min(len(portfolio_values), len(portfolio_index))
            portfolio_values = portfolio_values[:min_len]
            portfolio_index = portfolio_index[:min_len]
        
        portfolio_series = pd.Series(portfolio_values, index=portfolio_index)
        rolling_max = portfolio_series.expanding(min_periods=1).max()
        daily_drawdown = (portfolio_series / rolling_max) - 1.0
        mdd_percent = daily_drawdown.min() * 100

    # 승률 계산
    total_executed_trades = num_wins + num_losses
    win_rate = (num_wins / total_executed_trades) * 100 if total_executed_trades > 0 else 0

    # 손익비 계산
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') # 총이익/총손실
    if total_profit == 0 and total_loss == 0:
        profit_factor = 1 # 거래가 없거나 손익이 0인 경우

    print(f"\n백테스팅 완료:")
    print(f"  최종 자본: ${final_capital:,.2f}")
    print(f"  총 수익률: {total_return_percent:.2f}%")
    print(f"  최대 낙폭 (MDD): {mdd_percent:.2f}%")
    print(f"  총 거래 횟수 (매수/매도/손절): {num_trades}")
    print(f"  승리한 거래 수: {num_wins}")
    print(f"  패배한 거래 수: {num_losses}")
    print(f"  승률: {win_rate:.2f}%")
    print(f"  손익비 (Profit Factor): {profit_factor:.2f}")

    return {
        'total_return_percent': total_return_percent,
        'mdd_percent': mdd_percent,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'final_capital': final_capital,
        'num_trades': num_trades,
        'trade_log': pd.DataFrame(trade_log) if trade_log else pd.DataFrame(),
        'portfolio_values': portfolio_series
    }

# 메인 실행 로직
if __name__ == "__main__":
    # --- 하이퍼파라미터 및 설정 ---
    csv_file_path = 'bitcoin_processed_data.csv'
    feature_columns = ['Price_Change_Pct', 'Volume', 'RSI', 'MACD_Hist'] 
    target_column = 'NextDayPriceIncrease'
    
    sequence_length = 15          # LSTM 입력 시퀀스 길이 (더 줄여봄)
    hidden_dim = 128              # LSTM 히든 레이어 차원 (다시 늘림)
    num_layers = 3                # 레이어 수 증가
    dropout_rate_lstm = 0.3       # 드롭아웃 증가
    learning_rate = 0.0005        # 학습률 줄임
    epochs = 150                  # 훈련 에포크 수 더 증가
    batch_size = 64               # 배치 사이즈 증가
    # test_split_ratio = 0.2 # 고정 비율 대신 5년치 데이터 사용

    backtest_years = 5 # 최근 N년 데이터로 백테스팅
    stop_loss_pct_bt = 0.06       # 손절매 6%로 조정
    transaction_cost_pct_bt = 0.0005 # 0.05% 
    signal_thresh_buy_bt = 0.58   # 매수 임계값을 더 보수적으로 (0.52 -> 0.58)
    signal_thresh_sell_bt = 0.42  # 매도 임계값을 더 보수적으로 (0.48 -> 0.42)
    initial_capital_bt = 10000
    
    # --- 1. 데이터 로딩 ---
    if not os.path.exists(csv_file_path):
        print(f"오류: '{csv_file_path}' 파일을 찾을 수 없습니다.")
        exit()
        
    print(f"'{csv_file_path}' 파일에서 데이터 로딩 중...")
    df_full = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)

    if df_full.empty:
        print(f"오류: '{csv_file_path}' 파일이 비어있거나 로드할 수 없습니다.")
        exit()

    if 'GoogleTrends' in df_full.columns and not df_full['GoogleTrends'].isnull().all():
        if not df_full['GoogleTrends'].fillna(0).eq(0).all():
            feature_columns.append('GoogleTrends')
            print("GoogleTrends를 특징으로 포함합니다.")
        else:
            print("GoogleTrends 데이터가 모두 0 또는 NaN이므로 특징에 포함하지 않습니다.")
    else:
        print("GoogleTrends 컬럼이 없거나 모두 NaN이므로 특징에 포함하지 않습니다.")

    all_req_columns = feature_columns + [target_column] + ['Close', 'Open', 'High', 'Low'] # 백테스팅을 위해 OHLC 추가
    missing_cols = [col for col in all_req_columns if col not in df_full.columns]
    if missing_cols:
        print(f"오류: 필수 컬럼 부족: {missing_cols}")
        exit()

    df_processed_full = df_full[all_req_columns].copy()
    
    if df_processed_full.isnull().values.any():
        print("경고: 데이터에 NaN. ffill/bfill 처리.")
        df_processed_full = df_processed_full.ffill().bfill()
        df_processed_full.dropna(inplace=True) 
        if df_processed_full.empty:
            print("오류: NaN 처리 후 데이터 비어있음.")
            exit()
    
    # --- 2. 데이터 분할 (훈련 세트 / 테스트 세트 - 최근 5년) ---
    print(f"\n데이터 분할 중 (최근 {backtest_years}년 테스트)...")
    
    # 최근 5년 데이터의 시작일 계산
    if df_processed_full.empty:
        print("오류: 데이터프레임이 비어 분할할 수 없습니다.")
        exit()
        
    latest_date = df_processed_full.index.max()
    five_years_ago = latest_date - pd.DateOffset(years=backtest_years)
    
    df_train_val = df_processed_full[df_processed_full.index < five_years_ago]
    df_test_for_backtest = df_processed_full[df_processed_full.index >= five_years_ago]

    if df_train_val.empty or df_test_for_backtest.empty:
        print(f"오류: 데이터를 {backtest_years}년 기준으로 분할할 수 없습니다. 데이터 기간을 확인하세요.")
        print(f"전체 데이터 기간: {df_processed_full.index.min()} ~ {df_processed_full.index.max()}")
        print(f"훈련/검증 데이터 기간 (요청): < {five_years_ago}")
        print(f"테스트 데이터 기간 (요청): >= {five_years_ago}")
        print(f"실제 훈련/검증 데이터 수: {len(df_train_val)}, 실제 테스트 데이터 수: {len(df_test_for_backtest)}")
        exit()

    print(f"훈련/검증 데이터 기간: {df_train_val.index.min()} ~ {df_train_val.index.max()} (개수: {len(df_train_val)})")
    print(f"테스트(백테스트) 데이터 기간: {df_test_for_backtest.index.min()} ~ {df_test_for_backtest.index.max()} (개수: {len(df_test_for_backtest)})")

    # --- 3. 훈련 데이터 전처리 (스케일링, 시퀀스 생성) ---
    print("\n훈련 데이터 전처리 중...")
    scaler = StandardScaler()
    scaled_features_train_val_array = scaler.fit_transform(df_train_val[feature_columns])
    scaled_features_train_val_df = pd.DataFrame(scaled_features_train_val_array, columns=feature_columns, index=df_train_val.index)
    target_train_val_series = df_train_val[target_column]

    X_train_val_seq, y_train_val_seq = create_sequences(scaled_features_train_val_df, target_train_val_series, sequence_length)

    if len(X_train_val_seq) == 0:
        print("오류: 훈련 시퀀스 생성 실패.")
        exit()

    # 훈련/검증 데이터 분할 (시퀀스 데이터 대상)
    # 여기서는 간단히 훈련 데이터 전체를 사용 (검증 세트 분리 생략)
    X_train_tensor = torch.tensor(X_train_val_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_val_seq, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- 4. 테스트 데이터 전처리 (스케일링, 시퀀스 생성) ---
    # 테스트 데이터는 훈련 데이터에서 학습한 scaler를 그대로 사용 (transform만)
    print("\n테스트 데이터 전처리 중...")
    scaled_features_test_array = scaler.transform(df_test_for_backtest[feature_columns])
    scaled_features_test_df = pd.DataFrame(scaled_features_test_array, columns=feature_columns, index=df_test_for_backtest.index)
    target_test_series = df_test_for_backtest[target_column]

    X_test_seq, y_test_seq = create_sequences(scaled_features_test_df, target_test_series, sequence_length)
    
    if len(X_test_seq) == 0:
        print("오류: 테스트 시퀀스 생성 실패.")
        exit()
        
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)
    test_loader_for_eval = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # --- 5. 모델 초기화, 손실 함수, 옵티마이저 ---
    input_dim = X_train_val_seq.shape[2]
    model = LSTMPricePredictor(input_dim, hidden_dim, num_layers, output_dim=1, dropout_rate=dropout_rate_lstm).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"\nLSTM 모델 구조: \n{model}")

    # --- 6. 모델 훈련 ---
    train_model(model, train_loader, criterion, optimizer, epochs=epochs)

    # --- 7. 모델 분류 성능 평가 (테스트 세트) ---
    print("\n--- 모델 분류 성능 평가 (테스트 데이터) ---")
    evaluate_model(model, test_loader_for_eval, criterion) # 정의된 평가 함수 사용

    # --- 8. 백테스팅을 위한 예측 확률 생성 ---
    print("\n--- 백테스팅을 위한 예측 확률 생성 (테스트 데이터) ---")
    model.eval()
    all_test_predictions_proba = []
    # DataLoader를 사용하지 않고 전체 테스트 시퀀스에 대해 한 번에 예측 (메모리 주의)
    # 또는 배치 단위로 예측 후 합치기
    with torch.no_grad():
        # 전체 X_test_tensor를 모델에 전달
        if X_test_tensor.size(0) > 0:
            predictions_tensor = model(X_test_tensor.to(device))
            all_test_predictions_proba = predictions_tensor.cpu().squeeze().numpy()
        else:
            print("경고: X_test_tensor가 비어있어 예측 확률을 생성할 수 없습니다.")

    if len(all_test_predictions_proba) == 0:
        print("오류: 테스트 데이터에 대한 예측 확률 생성에 실패했습니다.")
        exit()

    # 예측 확률(all_test_predictions_proba)은 y_test_seq와 길이가 같음.
    # y_test_seq의 인덱스는 scaled_features_test_df에서 (sequence_length) 이후의 날짜들임.
    # 따라서, signals_series의 인덱스는 이 날짜들을 사용해야 함.
    # create_sequences에서 target.iloc[i + sequence_length]를 사용했으므로,
    # y_test_seq에 해당하는 원본 데이터의 인덱스는 df_test_for_backtest.index[sequence_length:] 임.
    
    prediction_dates = scaled_features_test_df.index[sequence_length : sequence_length + len(all_test_predictions_proba)]
    
    if len(prediction_dates) != len(all_test_predictions_proba):
        print(f"경고: 예측 날짜 길이({len(prediction_dates)})와 예측 확률 길이({len(all_test_predictions_proba)}) 불일치.")
        min_len = min(len(prediction_dates), len(all_test_predictions_proba))
        prediction_dates = prediction_dates[:min_len]
        all_test_predictions_proba = all_test_predictions_proba[:min_len]
        if min_len == 0:
            print("오류: 예측 날짜 또는 확률이 없어 백테스팅을 진행할 수 없습니다.")
            exit()
            
    signals_series_for_backtest = pd.Series(all_test_predictions_proba, index=prediction_dates)
    
    # --- 9. 백테스팅 실행 ---
    # df_test_for_backtest는 'Close', 'Open', 'High', 'Low' 컬럼을 포함해야 함
    # signals_series_for_backtest의 인덱스에 맞춰 df_test_for_backtest에서 가격 데이터를 가져옴.
    
    # signals_series_for_backtest가 비어 있는지 먼저 확인
    if signals_series_for_backtest.empty:
        print("오류: 생성된 신호 시리즈가 비어있어 백테스팅을 진행할 수 없습니다.")
        price_data_for_backtest = pd.DataFrame() # 빈 데이터프레임으로 초기화
    else:
        price_data_for_backtest = df_test_for_backtest.loc[signals_series_for_backtest.index, ['Close', 'Open', 'High', 'Low']]

    if price_data_for_backtest.empty or signals_series_for_backtest.empty:
        print("오류: 백테스팅을 위한 가격 데이터 또는 신호 데이터가 비어있습니다.")
        # backtest_results 변수가 정의되지 않을 수 있으므로, 기본값 설정 또는 프로그램 종료
        backtest_results = None 
    else:
        backtest_results = backtest_strategy_lstm(
            df_price_data=price_data_for_backtest,
            signals_series=signals_series_for_backtest,
            initial_capital=initial_capital_bt,
            stop_loss_pct=stop_loss_pct_bt,
            transaction_cost_pct=transaction_cost_pct_bt,
            signal_threshold_buy=signal_thresh_buy_bt,
            signal_threshold_sell=signal_thresh_sell_bt
        )
    
    if backtest_results: # 백테스팅이 성공적으로 실행된 경우에만 결과 출력
        print("\n--- 최종 백테스팅 결과 ---")
        # price_data_for_backtest가 비어있을 수 있는 경우를 대비
        if not price_data_for_backtest.empty:
            print(f"  테스트 기간: {price_data_for_backtest.index.min().strftime('%Y-%m-%d')} ~ {price_data_for_backtest.index.max().strftime('%Y-%m-%d')}")
        else:
            print("  테스트 기간: 정보 없음 (데이터 부족)")
        print(f"  총 수익률: {backtest_results['total_return_percent']:.2f}%")
        print(f"  최대 낙폭 (MDD): {backtest_results['mdd_percent']:.2f}%")
        print(f"  승률: {backtest_results['win_rate']:.2f}%")
        print(f"  손익비: {backtest_results['profit_factor']:.2f}")
        print(f"  최종 자본: ${backtest_results['final_capital']:,.2f}")
        print(f"  총 거래 횟수: {backtest_results['num_trades']}")
    else:
        print("\n백테스팅을 실행하지 못했거나 결과가 없습니다.")

    print("\n--- LSTM 주가 예측 및 백테스팅 프로그램 종료 ---")
