import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import random
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# GPU 사용 설정 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 시퀀스 데이터 생성 함수
def create_sequences(features, target, sequence_length=10):
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# 모델 훈련 함수 (간소화)
def train_model_simple(model, train_loader, criterion, optimizer, epochs=50):
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
    return epoch_loss / len(train_loader)

# 백테스팅 함수 (간소화)
def simple_backtest(df_price_data, signals_series, params):
    initial_capital = 10000
    capital = initial_capital
    holdings = 0
    entry_price = 0
    num_wins = 0
    num_losses = 0
    total_profit = 0
    total_loss = 0
    portfolio_values = []

    stop_loss_pct = params['stop_loss_pct']
    transaction_cost_pct = params['transaction_cost_pct']
    signal_threshold_buy = params['signal_thresh_buy']
    signal_threshold_sell = params['signal_thresh_sell']
    
    valid_price_data = df_price_data.loc[signals_series.index]
    
    if len(valid_price_data) != len(signals_series):
        common_index = signals_series.index.intersection(valid_price_data.index)
        signals_series = signals_series.loc[common_index]
        valid_price_data = valid_price_data.loc[common_index]
        if common_index.empty:
            return {'total_return_percent': -100, 'mdd_percent': -100, 'win_rate': 0, 'profit_factor': 0}

    for date, current_price in valid_price_data['Close'].items():
        current_signal_prob = signals_series.loc[date]
        action = 0

        if current_signal_prob > signal_threshold_buy:
            action = 1
        elif current_signal_prob < signal_threshold_sell:
            action = -1
        
        portfolio_values.append(capital + holdings * current_price)

        # 손절매
        if holdings > 0 and current_price < entry_price * (1 - stop_loss_pct):
            sell_amount = holdings
            proceeds = sell_amount * current_price * (1 - transaction_cost_pct)
            capital += proceeds
            
            trade_profit = proceeds - (sell_amount * entry_price * (1 + transaction_cost_pct))
            if trade_profit > 0: 
                num_wins += 1
                total_profit += trade_profit
            else: 
                num_losses += 1
                total_loss += abs(trade_profit)

            holdings = 0
            entry_price = 0
            action = 0

        # 거래 실행
        if action == 1 and capital > 0:
            # 신호 강도에 따른 투자 비율
            if current_signal_prob > 0.7:
                investment_ratio = 0.8
            elif current_signal_prob > 0.65:
                investment_ratio = 0.6
            else:
                investment_ratio = 0.4
                
            invest_amount = capital * investment_ratio
            buy_amount_in_currency = invest_amount / (1 + transaction_cost_pct)
            amount_to_buy = buy_amount_in_currency / current_price
            holdings += amount_to_buy
            capital -= invest_amount
            entry_price = current_price
            
        elif action == -1 and holdings > 0:
            sell_amount = holdings
            proceeds = sell_amount * current_price * (1 - transaction_cost_pct)
            capital += proceeds
            
            trade_profit = proceeds - (sell_amount * entry_price * (1 + transaction_cost_pct))
            if trade_profit > 0: 
                num_wins += 1
                total_profit += trade_profit
            else: 
                num_losses += 1
                total_loss += abs(trade_profit)

            holdings = 0
            entry_price = 0

    # 최종 청산
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

    final_capital = capital
    total_return_percent = (final_capital - initial_capital) / initial_capital * 100
    
    # MDD 계산
    if portfolio_values:
        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.expanding(min_periods=1).max()
        daily_drawdown = (portfolio_series / rolling_max) - 1.0
        mdd_percent = daily_drawdown.min() * 100
    else:
        mdd_percent = 0

    total_executed_trades = num_wins + num_losses
    win_rate = (num_wins / total_executed_trades) * 100 if total_executed_trades > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    if total_profit == 0 and total_loss == 0:
        profit_factor = 1

    return {
        'total_return_percent': total_return_percent,
        'mdd_percent': mdd_percent,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'final_capital': final_capital,
        'num_trades': total_executed_trades
    }

def generate_random_params():
    """랜덤한 하이퍼파라미터 조합 생성"""
    return {
        'sequence_length': random.choice([10, 15, 20, 25, 30]),
        'hidden_dim': random.choice([32, 64, 128, 256]),
        'num_layers': random.choice([1, 2, 3]),
        'dropout_rate': random.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        'learning_rate': random.choice([0.0001, 0.0005, 0.001, 0.002, 0.005]),
        'epochs': random.choice([30, 50, 70, 100, 150]),
        'batch_size': random.choice([16, 32, 64, 128]),
        'signal_thresh_buy': random.choice([0.55, 0.58, 0.6, 0.62, 0.65, 0.68, 0.7]),
        'signal_thresh_sell': random.choice([0.3, 0.32, 0.35, 0.38, 0.4, 0.42, 0.45]),
        'stop_loss_pct': random.choice([0.04, 0.05, 0.06, 0.08, 0.1, 0.12]),
        'transaction_cost_pct': 0.0005  # 고정값 (0.05%)
    }

def run_single_test(df_train_val, df_test_for_backtest, feature_columns, target_column, test_id):
    """단일 하이퍼파라미터 조합 테스트"""
    try:
        params = generate_random_params()
        
        print(f"\n=== 테스트 {test_id}/100 ===")
        print(f"파라미터: {params}")
        
        # 데이터 전처리
        scaler = StandardScaler()
        scaled_features_train_val_array = scaler.fit_transform(df_train_val[feature_columns])
        scaled_features_train_val_df = pd.DataFrame(scaled_features_train_val_array, columns=feature_columns, index=df_train_val.index)
        target_train_val_series = df_train_val[target_column]

        X_train_val_seq, y_train_val_seq = create_sequences(scaled_features_train_val_df, target_train_val_series, params['sequence_length'])

        if len(X_train_val_seq) == 0:
            return None
            
        X_train_tensor = torch.tensor(X_train_val_seq, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_val_seq, dtype=torch.float32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

        # 테스트 데이터 처리
        scaled_features_test_array = scaler.transform(df_test_for_backtest[feature_columns])
        scaled_features_test_df = pd.DataFrame(scaled_features_test_array, columns=feature_columns, index=df_test_for_backtest.index)
        target_test_series = df_test_for_backtest[target_column]

        X_test_seq, y_test_seq = create_sequences(scaled_features_test_df, target_test_series, params['sequence_length'])
        
        if len(X_test_seq) == 0:
            return None

        # 모델 생성 및 훈련
        input_dim = X_train_val_seq.shape[2]
        model = LSTMPricePredictor(input_dim, params['hidden_dim'], params['num_layers'], 
                                   output_dim=1, dropout_rate=params['dropout_rate']).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        # 모델 훈련
        final_loss = train_model_simple(model, train_loader, criterion, optimizer, params['epochs'])

        # 예측
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
            predictions_tensor = model(X_test_tensor)
            all_test_predictions_proba = predictions_tensor.cpu().squeeze().numpy()

        # 백테스팅
        prediction_dates = scaled_features_test_df.index[params['sequence_length'] : params['sequence_length'] + len(all_test_predictions_proba)]
        
        if len(prediction_dates) != len(all_test_predictions_proba):
            min_len = min(len(prediction_dates), len(all_test_predictions_proba))
            prediction_dates = prediction_dates[:min_len]
            all_test_predictions_proba = all_test_predictions_proba[:min_len]
            
        signals_series_for_backtest = pd.Series(all_test_predictions_proba, index=prediction_dates)
        price_data_for_backtest = df_test_for_backtest.loc[signals_series_for_backtest.index, ['Close', 'Open', 'High', 'Low']]

        backtest_results = simple_backtest(price_data_for_backtest, signals_series_for_backtest, params)
        
        result = {
            'test_id': test_id,
            'params': params,
            'final_loss': final_loss,
            'backtest_results': backtest_results,
            'score': backtest_results['total_return_percent']  # 최적화 목표
        }
        
        print(f"결과: 수익률 {backtest_results['total_return_percent']:.2f}%, MDD {backtest_results['mdd_percent']:.2f}%, 승률 {backtest_results['win_rate']:.2f}%")
        return result
        
    except Exception as e:
        print(f"테스트 {test_id} 실패: {e}")
        return None

def main():
    # 데이터 로딩
    csv_file_path = 'bitcoin_processed_data.csv'
    feature_columns = ['Price_Change_Pct', 'Volume', 'RSI', 'MACD_Hist']
    target_column = 'NextDayPriceIncrease'
    
    if not os.path.exists(csv_file_path):
        print(f"오류: '{csv_file_path}' 파일을 찾을 수 없습니다.")
        return
        
    print(f"'{csv_file_path}' 파일에서 데이터 로딩 중...")
    df_full = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)

    if 'GoogleTrends' in df_full.columns and not df_full['GoogleTrends'].isnull().all():
        if not df_full['GoogleTrends'].fillna(0).eq(0).all():
            feature_columns.append('GoogleTrends')
            print("GoogleTrends를 특징으로 포함합니다.")

    all_req_columns = feature_columns + [target_column] + ['Close', 'Open', 'High', 'Low']
    missing_cols = [col for col in all_req_columns if col not in df_full.columns]
    if missing_cols:
        print(f"오류: 필수 컬럼 부족: {missing_cols}")
        return

    df_processed_full = df_full[all_req_columns].copy()
    df_processed_full = df_processed_full.ffill().bfill()
    df_processed_full.dropna(inplace=True)
    
    # 데이터 분할
    backtest_years = 5
    latest_date = df_processed_full.index.max()
    five_years_ago = latest_date - pd.DateOffset(years=backtest_years)
    
    df_train_val = df_processed_full[df_processed_full.index < five_years_ago]
    df_test_for_backtest = df_processed_full[df_processed_full.index >= five_years_ago]

    print(f"훈련 데이터: {len(df_train_val)}개, 테스트 데이터: {len(df_test_for_backtest)}개")
    
    # 100번 테스트 실행
    all_results = []
    for i in range(1, 101):
        result = run_single_test(df_train_val, df_test_for_backtest, feature_columns, target_column, i)
        if result:
            all_results.append(result)
    
    # 결과 정렬 및 저장
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'hyperparameter_tuning_results_{timestamp}.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n=== 하이퍼파라미터 튜닝 완료 ===")
    print(f"총 {len(all_results)}개 결과가 {results_file}에 저장되었습니다.")
    
    # 상위 5개 결과 출력
    print("\n=== 상위 5개 결과 ===")
    for i, result in enumerate(all_results[:5]):
        br = result['backtest_results']
        print(f"\n{i+1}위:")
        print(f"  수익률: {br['total_return_percent']:.2f}%")
        print(f"  MDD: {br['mdd_percent']:.2f}%")
        print(f"  승률: {br['win_rate']:.2f}%")
        print(f"  손익비: {br['profit_factor']:.2f}")
        print(f"  거래수: {br['num_trades']}")
        print(f"  파라미터: {result['params']}")

if __name__ == "__main__":
    main() 