# XGBoost과 LightGBM 모델을 사용한 테더 코인 백테스트 및 성능 비교

import pyupbit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# XGBoost와 LightGBM 라이브러리
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_tether_data():
    """테더 코인 5년간 데이터 가져오기"""
    try:
        # 더 많은 데이터를 위해 count를 늘림
        print("테더 코인 데이터 수집 중...")
        
        # 최대한 많은 데이터 가져오기 (2000일 = 약 5.5년)
        df = pyupbit.get_ohlcv("KRW-USDT", interval="day", count=2000)
        
        if df is None or df.empty:
            print("데이터를 가져올 수 없습니다.")
            return None
            
        print(f"수집된 데이터 개수: {len(df)}개")
        print(f"데이터 기간: {df.index[0]} ~ {df.index[-1]}")
        
        return df
        
    except Exception as e:
        print(f"데이터 수집 중 오류 발생: {e}")
        return None

def create_features(df, lookback_days=30):
    """기술적 지표 및 특성 생성"""
    data = df.copy()
    
    # 기본 가격 특성
    data['price_change'] = data['close'].pct_change()
    data['high_low_ratio'] = data['high'] / data['low']
    data['volume_change'] = data['volume'].pct_change()
    
    # 이동평균
    for period in [5, 10, 20, 30]:
        data[f'ma_{period}'] = data['close'].rolling(window=period).mean()
        data[f'ma_ratio_{period}'] = data['close'] / data[f'ma_{period}']
    
    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    data['rsi'] = calculate_rsi(data['close'])
    
    # 볼린저 밴드
    data['bb_middle'] = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    # MACD
    exp1 = data['close'].ewm(span=12).mean()
    exp2 = data['close'].ewm(span=26).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_histogram'] = data['macd'] - data['macd_signal']
    
    # 변동성
    data['volatility'] = data['close'].rolling(window=20).std()
    
    # 과거 수익률
    for lag in [1, 2, 3, 5, 10]:
        data[f'return_lag_{lag}'] = data['close'].pct_change(lag)
    
    # 타겟 변수: 다음날 수익률
    data['target'] = data['close'].shift(-1) / data['close'] - 1
    
    return data

def prepare_ml_data(data, lookback_days=30):
    """머신러닝을 위한 데이터 준비"""
    # 특성 컬럼 선택
    feature_columns = [
        'price_change', 'high_low_ratio', 'volume_change',
        'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_30',
        'rsi', 'bb_position', 'macd', 'macd_signal', 'macd_histogram',
        'volatility', 'return_lag_1', 'return_lag_2', 'return_lag_3', 
        'return_lag_5', 'return_lag_10'
    ]
    
    # NaN 제거
    clean_data = data[feature_columns + ['target']].dropna()
    
    X = clean_data[feature_columns]
    y = clean_data['target']
    
    return X, y, clean_data.index

def train_models(X_train, y_train, X_val, y_val):
    """XGBoost와 LightGBM 모델 훈련"""
    
    # XGBoost 모델
    print("XGBoost 모델 훈련 중...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # LightGBM 모델
    print("LightGBM 모델 훈련 중...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # 검증 성능 평가
    xgb_pred = xgb_model.predict(X_val)
    lgb_pred = lgb_model.predict(X_val)
    
    xgb_mse = mean_squared_error(y_val, xgb_pred)
    lgb_mse = mean_squared_error(y_val, lgb_pred)
    
    xgb_mae = mean_absolute_error(y_val, xgb_pred)
    lgb_mae = mean_absolute_error(y_val, lgb_pred)
    
    print(f"\n=== 모델 검증 성능 ===")
    print(f"XGBoost - MSE: {xgb_mse:.6f}, MAE: {xgb_mae:.6f}")
    print(f"LightGBM - MSE: {lgb_mse:.6f}, MAE: {lgb_mae:.6f}")
    
    return xgb_model, lgb_model

def backtest_strategy(model, X_test, y_test, test_dates, model_name, initial_capital=1000000):
    """백테스트 실행"""
    
    predictions = model.predict(X_test)
    
    # 거래 신호 생성
    buy_threshold = 0.002   # 0.2% 이상 상승 예상시 매수
    sell_threshold = -0.001  # 0.1% 이상 하락 예상시 매도
    
    capital = initial_capital
    position = 0  # 0: 현금, 1: 보유
    trades = []
    portfolio_values = []
    
    for i in range(len(predictions)):
        current_date = test_dates[i]
        pred_return = predictions[i]
        actual_return = y_test.iloc[i]
        
        # 거래 신호
        if position == 0 and pred_return > buy_threshold:  # 매수
            position = 1
            buy_price = capital
            trades.append({
                'date': current_date,
                'action': 'BUY',
                'predicted_return': pred_return,
                'actual_return': actual_return,
                'capital': capital
            })
        elif position == 1 and (pred_return < sell_threshold or actual_return > 0.02):  # 매도 (손절 또는 익절)
            position = 0
            capital = capital * (1 + actual_return)
            trades.append({
                'date': current_date,
                'action': 'SELL',
                'predicted_return': pred_return,
                'actual_return': actual_return,
                'capital': capital
            })
        elif position == 1:  # 보유 중
            capital = capital * (1 + actual_return)
        
        portfolio_values.append(capital)
    
    # 성과 지표 계산
    total_return = (capital - initial_capital) / initial_capital * 100
    
    # 연수익률 (CAGR)
    years = len(X_test) / 365
    cagr = (capital / initial_capital) ** (1/years) - 1 if years > 0 else 0
    
    # 최대낙폭 (MDD)
    portfolio_series = pd.Series(portfolio_values)
    rolling_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series - rolling_max) / rolling_max
    mdd = drawdown.min()
    
    # 샤프 지수
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # 승률
    winning_trades = [t for t in trades if t['action'] == 'SELL' and t['actual_return'] > 0]
    total_trades = len([t for t in trades if t['action'] == 'SELL'])
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    
    results = {
        'model_name': model_name,
        'total_return': total_return,
        'cagr': cagr * 100,
        'mdd': mdd * 100,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'final_capital': capital,
        'portfolio_values': portfolio_values,
        'trades': trades,
        'test_dates': test_dates
    }
    
    return results

def plot_results(xgb_results, lgb_results, tether_data):
    """결과 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 포트폴리오 가치 변화
    ax1 = axes[0, 0]
    ax1.plot(xgb_results['test_dates'], xgb_results['portfolio_values'], 
             label=f"XGBoost (수익률: {xgb_results['total_return']:.2f}%)", linewidth=2)
    ax1.plot(lgb_results['test_dates'], lgb_results['portfolio_values'], 
             label=f"LightGBM (수익률: {lgb_results['total_return']:.2f}%)", linewidth=2)
    ax1.axhline(y=1000000, color='gray', linestyle='--', alpha=0.7, label='초기 자본')
    ax1.set_title('포트폴리오 가치 변화', fontsize=14, fontweight='bold')
    ax1.set_xlabel('날짜')
    ax1.set_ylabel('포트폴리오 가치 (원)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 테더 가격 차트
    ax2 = axes[0, 1]
    test_start_idx = len(tether_data) - len(xgb_results['portfolio_values'])
    test_data = tether_data.iloc[test_start_idx:]
    ax2.plot(test_data.index, test_data['close'], color='orange', linewidth=2)
    ax2.set_title('테더 코인 가격 (테스트 기간)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('날짜')
    ax2.set_ylabel('가격 (원)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 성과 지표 비교
    ax3 = axes[1, 0]
    metrics = ['CAGR (%)', 'MDD (%)', 'Sharpe Ratio', 'Win Rate (%)']
    xgb_values = [xgb_results['cagr'], abs(xgb_results['mdd']), 
                  xgb_results['sharpe_ratio'], xgb_results['win_rate']]
    lgb_values = [lgb_results['cagr'], abs(lgb_results['mdd']), 
                  lgb_results['sharpe_ratio'], lgb_results['win_rate']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, xgb_values, width, label='XGBoost', alpha=0.8)
    bars2 = ax3.bar(x + width/2, lgb_values, width, label='LightGBM', alpha=0.8)
    
    ax3.set_title('성과 지표 비교', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # 4. 거래 횟수 및 기타 정보
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    info_text = f"""
    === XGBoost 결과 ===
    총 수익률: {xgb_results['total_return']:.2f}%
    연수익률 (CAGR): {xgb_results['cagr']:.2f}%
    최대낙폭 (MDD): {xgb_results['mdd']:.2f}%
    샤프 지수: {xgb_results['sharpe_ratio']:.2f}
    승률: {xgb_results['win_rate']:.2f}%
    총 거래 횟수: {xgb_results['total_trades']}회
    최종 자본: {xgb_results['final_capital']:,.0f}원
    
    === LightGBM 결과 ===
    총 수익률: {lgb_results['total_return']:.2f}%
    연수익률 (CAGR): {lgb_results['cagr']:.2f}%
    최대낙폭 (MDD): {lgb_results['mdd']:.2f}%
    샤프 지수: {lgb_results['sharpe_ratio']:.2f}
    승률: {lgb_results['win_rate']:.2f}%
    총 거래 횟수: {lgb_results['total_trades']}회
    최종 자본: {lgb_results['final_capital']:,.0f}원
    """
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='Malgun Gothic',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    """메인 실행 함수"""
    print("=== 테더 코인 XGBoost vs LightGBM 백테스트 ===\n")
    
    # 1. 데이터 수집
    print("1. 테더 코인 데이터 수집 중...")
    tether_data = get_tether_data()
    if tether_data is None:
        return
    
    # 2. 특성 생성
    print("2. 기술적 지표 및 특성 생성 중...")
    data_with_features = create_features(tether_data)
    
    # 3. 머신러닝 데이터 준비
    print("3. 머신러닝 데이터 준비 중...")
    X, y, dates = prepare_ml_data(data_with_features)
    
    print(f"총 데이터 포인트: {len(X)}개")
    print(f"특성 개수: {X.shape[1]}개")
    
    # 4. 데이터 분할 (사용 가능한 데이터에 따라 테스트 기간 조정)
    total_data = len(X)
    print(f"사용 가능한 총 데이터: {total_data}개")
    
    # 최소 훈련 데이터 확보를 위해 테스트 기간 조정
    if total_data >= 365:
        test_days = 365  # 12개월
    elif total_data >= 180:
        test_days = 180  # 6개월
    elif total_data >= 90:
        test_days = 90   # 3개월
    else:
        test_days = max(30, int(total_data * 0.2))  # 최소 30일 또는 20%
    
    print(f"테스트 기간: {test_days}일")
    
    # 훈련/검증 데이터 확보
    train_val_size = total_data - test_days
    
    # 검증 데이터가 최소 10개는 되도록 조정
    min_val_size = 10
    if train_val_size < min_val_size * 2:
        # 데이터가 너무 적으면 테스트 기간을 줄임
        test_days = max(10, total_data - min_val_size * 2)
        train_val_size = total_data - test_days
    
    train_size = train_val_size - min_val_size
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    
    X_val = X.iloc[train_size:train_val_size]
    y_val = y.iloc[train_size:train_val_size]
    
    X_test = X.iloc[train_val_size:]
    y_test = y.iloc[train_val_size:]
    test_dates = dates[train_val_size:]
    
    print(f"훈련 데이터: {len(X_train)}개")
    print(f"검증 데이터: {len(X_val)}개")
    print(f"테스트 데이터: {len(X_test)}개")
    
    # 5. 모델 훈련
    print("\n4. 모델 훈련 중...")
    xgb_model, lgb_model = train_models(X_train, y_train, X_val, y_val)
    
    # 6. 백테스트 실행
    print("\n5. 백테스트 실행 중...")
    print("XGBoost 백테스트...")
    xgb_results = backtest_strategy(xgb_model, X_test, y_test, test_dates, "XGBoost")
    
    print("LightGBM 백테스트...")
    lgb_results = backtest_strategy(lgb_model, X_test, y_test, test_dates, "LightGBM")
    
    # 7. 결과 출력
    print("\n=== 백테스트 결과 ===")
    print(f"\nXGBoost:")
    print(f"  총 수익률: {xgb_results['total_return']:.2f}%")
    print(f"  연수익률 (CAGR): {xgb_results['cagr']:.2f}%")
    print(f"  최대낙폭 (MDD): {xgb_results['mdd']:.2f}%")
    print(f"  샤프 지수: {xgb_results['sharpe_ratio']:.2f}")
    print(f"  승률: {xgb_results['win_rate']:.2f}%")
    print(f"  총 거래 횟수: {xgb_results['total_trades']}회")
    
    print(f"\nLightGBM:")
    print(f"  총 수익률: {lgb_results['total_return']:.2f}%")
    print(f"  연수익률 (CAGR): {lgb_results['cagr']:.2f}%")
    print(f"  최대낙폭 (MDD): {lgb_results['mdd']:.2f}%")
    print(f"  샤프 지수: {lgb_results['sharpe_ratio']:.2f}")
    print(f"  승률: {lgb_results['win_rate']:.2f}%")
    print(f"  총 거래 횟수: {lgb_results['total_trades']}회")
    
    # 8. 결과 시각화
    print("\n6. 결과 시각화 중...")
    plot_results(xgb_results, lgb_results, tether_data)
    
    # 9. 성능 비교 요약
    print("\n=== 성능 비교 요약 ===")
    if xgb_results['total_return'] > lgb_results['total_return']:
        print(f"🏆 XGBoost가 더 높은 수익률을 달성했습니다!")
        print(f"   XGBoost: {xgb_results['total_return']:.2f}% vs LightGBM: {lgb_results['total_return']:.2f}%")
    else:
        print(f"🏆 LightGBM이 더 높은 수익률을 달성했습니다!")
        print(f"   LightGBM: {lgb_results['total_return']:.2f}% vs XGBoost: {xgb_results['total_return']:.2f}%")
    
    if abs(xgb_results['mdd']) < abs(lgb_results['mdd']):
        print(f"🛡️ XGBoost가 더 낮은 최대낙폭을 보였습니다!")
        print(f"   XGBoost: {xgb_results['mdd']:.2f}% vs LightGBM: {lgb_results['mdd']:.2f}%")
    else:
        print(f"🛡️ LightGBM이 더 낮은 최대낙폭을 보였습니다!")
        print(f"   LightGBM: {lgb_results['mdd']:.2f}% vs XGBoost: {xgb_results['mdd']:.2f}%")

if __name__ == "__main__":
    main()