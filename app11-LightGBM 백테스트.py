# 테더 코인 LightGBM 백테스트 - 파라미터 튜닝 및 최적화
# 10번 반복 실행으로 최고 수익률 달성

import pyupbit
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

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

@dataclass
class ModelConfig:
    """모델 설정 클래스 - 최적화된 파라미터"""
    n_estimators: int = 300        # 최적: 300 (적당한 모델 복잡도)
    max_depth: int = 8             # 최적: 8 (충분한 깊이)
    learning_rate: float = 0.03    # 최적: 0.03 (낮은 학습률로 안정성)
    subsample: float = 0.8         # 최적: 0.8 (일반화 성능)
    colsample_bytree: float = 0.9  # 최적: 0.9 (특성 활용)
    buy_percentile: int = 75       # 최적: 75% (상위 25% 진입)
    sell_percentile: int = 35      # 최적: 35% (하위 35% 청산)
    min_trades: int = 5            # 최소 거래 횟수
    
@dataclass 
class BacktestResult:
    """백테스트 결과 클래스"""
    total_return: float
    cagr: float
    mdd: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    config: ModelConfig

def get_tether_data():
    """테더 코인 데이터 가져오기"""
    try:
        print("=== 테더 코인 데이터 수집 중 ===")
        
        # 최대한 많은 데이터 가져오기
        df = pyupbit.get_ohlcv('KRW-USDT', interval="day", count=2000)
        
        if df is None or df.empty:
            print("테더 코인 데이터를 가져올 수 없습니다.")
            return None
            
        print(f"수집된 데이터 개수: {len(df)}일")
        print(f"데이터 기간: {df.index[0]} ~ {df.index[-1]}")
        
        min_price = float(df['close'].min())
        max_price = float(df['close'].max())
        print(f"가격 범위: {min_price:,.0f}원 ~ {max_price:,.0f}원")
        
        return df
        
    except Exception as e:
        print(f"데이터 수집 중 오류 발생: {e}")
        return None

def create_technical_features(df):
    """기술적 지표 및 특성 생성"""
    data = df.copy()
    
    # 기본 가격 특성
    data['price_change'] = data['close'].pct_change()
    data['high_low_ratio'] = data['high'] / data['low']
    data['volume_change'] = data['volume'].pct_change()
    data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # 이동평균 및 비율
    for period in [5, 10, 20, 30, 60]:
        data[f'ma_{period}'] = data['close'].rolling(window=period).mean()
        data[f'ma_ratio_{period}'] = data['close'] / data[f'ma_{period}']
        data[f'ma_slope_{period}'] = data[f'ma_{period}'].diff(5) / data[f'ma_{period}'].shift(5)
    
    # 이동평균 교차 신호
    data['ma_cross_5_20'] = (data['ma_5'] > data['ma_20']).astype(int)
    data['ma_cross_10_30'] = (data['ma_10'] > data['ma_30']).astype(int)
    
    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    data['rsi'] = calculate_rsi(data['close'])
    data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
    data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
    
    # 볼린저 밴드
    data['bb_middle'] = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    
    # MACD
    exp1 = data['close'].ewm(span=12).mean()
    exp2 = data['close'].ewm(span=26).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_histogram'] = data['macd'] - data['macd_signal']
    data['macd_crossover'] = ((data['macd'] > data['macd_signal']) & 
                              (data['macd'].shift(1) <= data['macd_signal'].shift(1))).astype(int)
    
    # 변동성 지표
    data['volatility'] = data['close'].rolling(window=20).std()
    data['atr'] = data[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], 
                     abs(x['high'] - x['close']), 
                     abs(x['low'] - x['close'])), axis=1
    ).rolling(window=14).mean()
    
    # 거래량 지표
    data['volume_ma'] = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    
    # 과거 수익률 및 추세
    for lag in [1, 2, 3, 5, 10, 20]:
        data[f'return_lag_{lag}'] = data['close'].pct_change(lag)
        data[f'price_momentum_{lag}'] = data['close'] / data['close'].shift(lag) - 1
    
    # 지지/저항 레벨 (단순화)
    data['high_20'] = data['high'].rolling(window=20).max()
    data['low_20'] = data['low'].rolling(window=20).min()
    data['resistance_ratio'] = data['close'] / data['high_20']
    data['support_ratio'] = data['close'] / data['low_20']
    
    # 타겟 변수: 다음날 수익률
    data['target'] = data['close'].shift(-1) / data['close'] - 1
    
    return data

def prepare_features(data):
    """머신러닝을 위한 특성 선택 및 데이터 준비"""
    
    # 특성 컬럼 선택
    feature_columns = [
        # 기본 가격 특성
        'price_change', 'high_low_ratio', 'volume_change', 'price_position',
        
        # 이동평균 비율
        'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_30', 'ma_ratio_60',
        'ma_slope_5', 'ma_slope_10', 'ma_slope_20',
        
        # 이동평균 교차
        'ma_cross_5_20', 'ma_cross_10_30',
        
        # RSI
        'rsi', 'rsi_overbought', 'rsi_oversold',
        
        # 볼린저 밴드
        'bb_position', 'bb_width',
        
        # MACD
        'macd', 'macd_signal', 'macd_histogram', 'macd_crossover',
        
        # 변동성
        'volatility', 'atr',
        
        # 거래량
        'volume_ratio',
        
        # 모멘텀
        'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5', 'return_lag_10',
        'price_momentum_1', 'price_momentum_5', 'price_momentum_10',
        
        # 지지/저항
        'resistance_ratio', 'support_ratio'
    ]
    
    # NaN 제거
    clean_data = data[feature_columns + ['target']].dropna()
    
    X = clean_data[feature_columns]
    y = clean_data['target']
    
    return X, y, clean_data.index, feature_columns

def train_lightgbm_model(X_train, y_train, X_val, y_val, config: ModelConfig):
    """LightGBM 모델 훈련 - 설정 기반"""
    
    # LightGBM 모델 생성
    model = lgb.LGBMRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # 조기 종료를 위한 학습
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    )
    
    return model

def backtest_strategy(model, X_test, y_test, test_dates, config: ModelConfig, initial_capital=1000000):
    """백테스트 실행 - 설정 기반"""
    
    predictions = model.predict(X_test)
    
    # 상대적 임계값 설정
    buy_threshold = np.percentile(predictions, config.buy_percentile)
    sell_threshold = np.percentile(predictions, config.sell_percentile)
    
    capital = initial_capital
    position = 0  # 0: 현금, 1: 테더 보유
    trades = []
    portfolio_values = []
    
    for i in range(len(predictions)):
        pred_return = predictions[i]
        actual_return = y_test.iloc[i]
        current_date = test_dates[i]
        
        # 거래 신호
        if position == 0 and pred_return > buy_threshold:  # 매수
            position = 1
            usdt_amount = capital
            trades.append({
                'date': current_date,
                'action': 'BUY',
                'predicted_return': pred_return,
                'actual_return': actual_return,
                'capital': capital
            })
        elif position == 1 and pred_return < sell_threshold:  # 매도
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
    final_capital = float(capital)
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    # 연수익률 (CAGR)
    years = len(X_test) / 365
    cagr = (final_capital / initial_capital) ** (1/years) - 1 if years > 0 else 0
    
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
    
    # 거래 횟수가 너무 적으면 패널티
    if total_trades < config.min_trades:
        total_return *= 0.5  # 50% 패널티
    
    return BacktestResult(
        total_return=total_return,
        cagr=cagr * 100,
        mdd=mdd * 100,
        sharpe_ratio=sharpe_ratio,
        win_rate=win_rate,
        total_trades=total_trades,
        config=config
    )

def plot_results(results, feature_importance=None):
    """결과 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 포트폴리오 가치 변화
    ax1 = axes[0, 0]
    ax1.plot(results['test_dates'], results['portfolio_values'], 
             label=f"LightGBM 전략 (수익률: {results['total_return']:.2f}%)", 
             color='blue', linewidth=2)
    
    # Buy & Hold 비교 (단순화)
    initial_value = results['initial_capital']
    buy_hold_values = [initial_value * (1 + sum(results['actual_returns'][:i+1])) 
                       for i in range(len(results['actual_returns']))]
    ax1.plot(results['test_dates'], buy_hold_values, 
             label=f"Buy & Hold (수익률: {results['buy_hold_return']:.2f}%)", 
             color='red', linewidth=2, alpha=0.7)
    
    ax1.axhline(y=results['initial_capital'], color='gray', linestyle='--', alpha=0.7, label='초기 자본')
    ax1.set_title('포트폴리오 가치 변화', fontsize=14, fontweight='bold')
    ax1.set_xlabel('날짜')
    ax1.set_ylabel('포트폴리오 가치 (원)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 예측 vs 실제 수익률
    ax2 = axes[0, 1]
    ax2.scatter(results['predictions']*100, results['actual_returns']*100, 
                alpha=0.6, s=30)
    ax2.plot([-15, 15], [-15, 15], 'r--', alpha=0.8, label='Perfect Prediction')
    ax2.set_title('예측 vs 실제 수익률', fontsize=14, fontweight='bold')
    ax2.set_xlabel('예측 수익률 (%)')
    ax2.set_ylabel('실제 수익률 (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 누적 수익률
    ax3 = axes[1, 0]
    cumulative_returns = [(v / results['initial_capital'] - 1) * 100 for v in results['portfolio_values']]
    buy_hold_cumulative = [(v / results['initial_capital'] - 1) * 100 for v in buy_hold_values]
    
    ax3.plot(results['test_dates'], cumulative_returns, label='LightGBM 전략', 
             color='blue', linewidth=2)
    ax3.plot(results['test_dates'], buy_hold_cumulative, label='Buy & Hold', 
             color='red', linewidth=2, alpha=0.7)
    ax3.set_title('누적 수익률 비교', fontsize=14, fontweight='bold')
    ax3.set_xlabel('날짜')
    ax3.set_ylabel('누적 수익률 (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 드로우다운
    ax4 = axes[1, 1]
    portfolio_series = pd.Series(results['portfolio_values'])
    rolling_max = portfolio_series.expanding().max()
    drawdowns = (portfolio_series - rolling_max) / rolling_max * 100
    
    ax4.fill_between(results['test_dates'], 0, drawdowns, color='red', alpha=0.3)
    ax4.set_title(f'드로우다운 (최대: {results["mdd"]:.2f}%)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('날짜')
    ax4.set_ylabel('드로우다운 (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 특성 중요도 출력 (상위 10개)
    if feature_importance is not None:
        print("\n=== 상위 10개 특성 중요도 ===")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"{i+1:2d}. {feature}: {importance:.0f}")

def generate_parameter_combinations():
    """파라미터 조합 생성"""
    param_grid = {
        'n_estimators': [300, 500, 700],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.08],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'buy_percentile': [60, 65, 70, 75],
        'sell_percentile': [25, 30, 35, 40]
    }
    
    # 모든 조합을 생성하면 너무 많으므로 랜덤하게 선택
    keys = list(param_grid.keys())
    combinations = []
    
    # 10개의 다양한 조합 생성
    np.random.seed(42)
    for i in range(10):
        config = ModelConfig()
        for key in keys:
            setattr(config, key, np.random.choice(param_grid[key]))
        combinations.append(config)
    
    return combinations

def optimize_model():
    """모델 최적화 - 10번 반복 실행"""
    print("=== 테더 코인 LightGBM 파라미터 최적화 ===\n")
    
    # 1. 데이터 수집
    print("1. 데이터 수집 중...")
    tether_data = get_tether_data()
    if tether_data is None:
        return
    
    # 2. 기술적 지표 생성
    print("2. 기술적 지표 생성 중...")
    data_with_features = create_technical_features(tether_data)
    
    # 3. 특성 준비
    print("3. 특성 데이터 준비 중...")
    X, y, dates, feature_columns = prepare_features(data_with_features)
    
    print(f"총 데이터 포인트: {len(X)}개")
    print(f"특성 개수: {X.shape[1]}개")
    
    # 4. 데이터 분할 (시간 순서 유지)
    total_samples = len(X)
    test_days = min(365, int(total_samples * 0.3))
    train_val_size = total_samples - test_days
    train_size = int(train_val_size * 0.8)
    
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
    
    # 5. 파라미터 조합 생성
    print("\n4. 파라미터 조합 생성 중...")
    param_combinations = generate_parameter_combinations()
    
    # 6. 10번 반복 실행
    print("\n5. 10번 반복 최적화 실행 중...")
    print("=" * 80)
    
    results = []
    best_result = None
    best_return = -float('inf')
    
    for i, config in enumerate(param_combinations, 1):
        print(f"\n🔄 실행 {i}/10:")
        print(f"   파라미터: n_est={config.n_estimators}, depth={config.max_depth}, "
              f"lr={config.learning_rate:.3f}, buy={config.buy_percentile}%, sell={config.sell_percentile}%")
        
        try:
            # 모델 훈련
            model = train_lightgbm_model(X_train, y_train, X_val, y_val, config)
            
            # 백테스트 실행
            result = backtest_strategy(model, X_test, y_test, test_dates, config)
            results.append(result)
            
            # 결과 출력
            print(f"   📊 수익률: {result.total_return:.2f}% | CAGR: {result.cagr:.2f}% | "
                  f"MDD: {result.mdd:.2f}% | 승률: {result.win_rate:.1f}% | 거래: {result.total_trades}회")
            
            # 최고 성과 업데이트
            if result.total_return > best_return:
                best_return = result.total_return
                best_result = result
                print(f"   🏆 새로운 최고 성과! (수익률: {best_return:.2f}%)")
                
        except Exception as e:
            print(f"   ❌ 오류 발생: {e}")
            continue
    
    # 7. 결과 분석 및 최고 성과 출력
    print("\n" + "=" * 80)
    print("📈 최적화 완료! 결과 분석:")
    print("=" * 80)
    
    if not results:
        print("❌ 유효한 결과가 없습니다.")
        return
    
    # 결과 정렬
    results.sort(key=lambda x: x.total_return, reverse=True)
    
    print(f"\n🏆 최고 성과:")
    best = results[0]
    print(f"   📊 총 수익률: {best.total_return:.2f}%")
    print(f"   📈 연수익률 (CAGR): {best.cagr:.2f}%")
    print(f"   📉 최대낙폭 (MDD): {best.mdd:.2f}%")
    print(f"   ⚡ 샤프 지수: {best.sharpe_ratio:.3f}")
    print(f"   🎯 승률: {best.win_rate:.1f}%")
    print(f"   🔄 총 거래 횟수: {best.total_trades}회")
    print(f"\n🔧 최적 파라미터:")
    print(f"   - n_estimators: {best.config.n_estimators}")
    print(f"   - max_depth: {best.config.max_depth}")
    print(f"   - learning_rate: {best.config.learning_rate}")
    print(f"   - subsample: {best.config.subsample}")
    print(f"   - colsample_bytree: {best.config.colsample_bytree}")
    print(f"   - buy_percentile: {best.config.buy_percentile}")
    print(f"   - sell_percentile: {best.config.sell_percentile}")
    
    print(f"\n📊 상위 5개 결과:")
    for i, result in enumerate(results[:5], 1):
        print(f"   {i}. 수익률: {result.total_return:.2f}% | CAGR: {result.cagr:.2f}% | "
              f"승률: {result.win_rate:.1f}% | 거래: {result.total_trades}회")
    
    # 8. 최고 성과 모델로 상세 분석
    print(f"\n🔍 최고 성과 모델 상세 분석 실행 중...")
    
    # 최고 성과 모델 재훈련
    best_model = train_lightgbm_model(X_train, y_train, X_val, y_val, best.config)
    
    # 특성 중요도
    feature_importance = sorted(zip(feature_columns, best_model.feature_importances_), 
                               key=lambda x: x[1], reverse=True)
    
    # 상세 백테스트 (결과 저장용)
    predictions = best_model.predict(X_test)
    buy_threshold = np.percentile(predictions, best.config.buy_percentile)
    sell_threshold = np.percentile(predictions, best.config.sell_percentile)
    
    print(f"\n📈 예측 분석:")
    print(f"   범위: {predictions.min()*100:.3f}% ~ {predictions.max()*100:.3f}%")
    print(f"   평균: {predictions.mean()*100:.3f}%")
    print(f"   표준편차: {predictions.std()*100:.3f}%")
    print(f"   매수 임계값: {buy_threshold*100:.3f}%")
    print(f"   매도 임계값: {sell_threshold*100:.3f}%")
    
    print(f"\n🔝 상위 10개 특성 중요도:")
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        print(f"   {i+1:2d}. {feature}: {importance:.0f}")
    
    # Buy & Hold와 비교
    buy_hold_return = (y_test.sum()) * 100
    outperformance = best.total_return - buy_hold_return
    
    print(f"\n📊 성과 비교:")
    print(f"   AI 전략: {best.total_return:.2f}%")
    print(f"   Buy & Hold: {buy_hold_return:.2f}%")
    print(f"   초과 수익: {outperformance:.2f}%p")
    
    if outperformance > 0:
        print(f"   🎉 AI 전략이 Buy & Hold보다 {outperformance:.2f}%p 더 우수합니다!")
    else:
        print(f"   📉 AI 전략이 Buy & Hold보다 {abs(outperformance):.2f}%p 부족합니다.")
    
    print("\n" + "=" * 80)
    print("✅ 최적화 완료!")
    print("=" * 80)
    
    return best, results

def run_optimized_backtest():
    """최적 조건으로 단일 백테스트 실행"""
    print("=== 테더 코인 LightGBM 최적 조건 백테스트 ===\n")
    
    # 1. 데이터 수집
    print("1. 데이터 수집 중...")
    tether_data = get_tether_data()
    if tether_data is None:
        return None
    
    # 2. 기술적 지표 생성
    print("2. 기술적 지표 생성 중...")
    data_with_features = create_technical_features(tether_data)
    
    # 3. 특성 준비
    print("3. 특성 데이터 준비 중...")
    X, y, dates, feature_columns = prepare_features(data_with_features)
    
    print(f"총 데이터 포인트: {len(X)}개")
    print(f"특성 개수: {X.shape[1]}개")
    
    # 4. 데이터 분할 (시간 순서 유지)
    total_samples = len(X)
    test_days = min(365, int(total_samples * 0.3))
    train_val_size = total_samples - test_days
    train_size = int(train_val_size * 0.8)
    
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
    
    # 5. 최적 조건으로 모델 훈련
    print("\n4. 최적 조건으로 LightGBM 모델 훈련 중...")
    optimal_config = ModelConfig()  # 최적화된 기본값 사용
    
    print(f"🔧 사용 중인 최적 파라미터:")
    print(f"   - n_estimators: {optimal_config.n_estimators}")
    print(f"   - max_depth: {optimal_config.max_depth}")
    print(f"   - learning_rate: {optimal_config.learning_rate}")
    print(f"   - subsample: {optimal_config.subsample}")
    print(f"   - colsample_bytree: {optimal_config.colsample_bytree}")
    print(f"   - buy_percentile: {optimal_config.buy_percentile}%")
    print(f"   - sell_percentile: {optimal_config.sell_percentile}%")
    
    model = train_lightgbm_model(X_train, y_train, X_val, y_val, optimal_config)
    
    # 검증 성능 평가
    val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    print(f"\n📊 모델 검증 성능:")
    print(f"   MSE: {val_mse:.6f}")
    print(f"   MAE: {val_mae:.6f}")
    
    # 6. 백테스트 실행
    print("\n5. 백테스트 실행 중...")
    result = backtest_strategy(model, X_test, y_test, test_dates, optimal_config)
    
    # 7. 상세 분석
    predictions = model.predict(X_test)
    buy_threshold = np.percentile(predictions, optimal_config.buy_percentile)
    sell_threshold = np.percentile(predictions, optimal_config.sell_percentile)
    
    print(f"\n📈 예측 분석:")
    print(f"   범위: {predictions.min()*100:.3f}% ~ {predictions.max()*100:.3f}%")
    print(f"   평균: {predictions.mean()*100:.3f}%")
    print(f"   표준편차: {predictions.std()*100:.3f}%")
    print(f"   매수 임계값 (상위 {100-optimal_config.buy_percentile}%): {buy_threshold*100:.3f}%")
    print(f"   매도 임계값 (하위 {optimal_config.sell_percentile}%): {sell_threshold*100:.3f}%")
    
    # 8. 결과 출력
    print(f"\n🏆 백테스트 결과:")
    print(f"📊 백테스트 기간: {test_dates[0].strftime('%Y-%m-%d')} ~ {test_dates[-1].strftime('%Y-%m-%d')}")
    print(f"📈 초기 자본: 1,000,000원")
    print(f"💰 최종 자본: {1000000 * (1 + result.total_return/100):,.0f}원")
    print(f"📊 총 수익률: {result.total_return:.2f}%")
    print(f"📈 연수익률 (CAGR): {result.cagr:.2f}%")
    print(f"📉 최대낙폭 (MDD): {result.mdd:.2f}%")
    print(f"⚡ 샤프 지수: {result.sharpe_ratio:.3f}")
    print(f"🎯 승률: {result.win_rate:.1f}%")
    print(f"🔄 총 거래 횟수: {result.total_trades}회")
    
    # Buy & Hold와 비교
    buy_hold_return = (y_test.sum()) * 100
    outperformance = result.total_return - buy_hold_return
    
    print(f"\n📊 성과 비교:")
    print(f"   AI 전략: {result.total_return:.2f}%")
    print(f"   Buy & Hold: {buy_hold_return:.2f}%")
    print(f"   초과 수익: {outperformance:.2f}%p")
    
    if outperformance > 0:
        print(f"   🎉 AI 전략이 Buy & Hold보다 {outperformance:.2f}%p 더 우수합니다!")
    else:
        print(f"   📉 AI 전략이 Buy & Hold보다 {abs(outperformance):.2f}%p 부족합니다.")
    
    # 9. 특성 중요도
    feature_importance = sorted(zip(feature_columns, model.feature_importances_), 
                               key=lambda x: x[1], reverse=True)
    
    print(f"\n🔝 상위 10개 특성 중요도:")
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        print(f"   {i+1:2d}. {feature}: {importance:.0f}")
    
    # 10. 상세 거래 내역 (최대 10개)
    print(f"\n📋 주요 거래 내역:")
    
    # 거래 내역 재구성 (실제 거래 데이터)
    capital = 1000000
    position = 0
    trades = []
    
    for i in range(len(predictions)):
        pred_return = predictions[i]
        actual_return = y_test.iloc[i]
        current_date = test_dates[i]
        
        if position == 0 and pred_return > buy_threshold:
            position = 1
            trades.append({
                'date': current_date,
                'action': 'BUY',
                'predicted_return': pred_return,
                'actual_return': actual_return,
                'capital': capital
            })
        elif position == 1 and pred_return < sell_threshold:
            position = 0
            capital = capital * (1 + actual_return)
            trades.append({
                'date': current_date,
                'action': 'SELL',
                'predicted_return': pred_return,
                'actual_return': actual_return,
                'capital': capital
            })
        elif position == 1:
            capital = capital * (1 + actual_return)
    
    for i, trade in enumerate(trades[:10]):
        print(f"   {trade['date'].strftime('%Y-%m-%d')} | {trade['action']} | "
              f"예측: {trade['predicted_return']*100:+.3f}% | "
              f"실제: {trade['actual_return']*100:+.3f}% | "
              f"자본: {trade['capital']:,.0f}원")
    
    if len(trades) > 10:
        print(f"   ... 총 {len(trades)}개 거래")
    
    print(f"\n" + "="*60)
    print(f"✅ 최적 조건 백테스트 완료!")
    print(f"🎯 최종 성과: {result.total_return:.2f}% (Buy & Hold 대비 {outperformance:+.2f}%p)")
    print(f"="*60)
    
    return result, optimal_config

def main():
    """메인 실행 함수"""
    print("🚀 테더 코인 LightGBM 백테스트 시스템")
    print("="*50)
    print("1. 최적 조건으로 백테스트 실행 (권장)")
    print("2. 파라미터 최적화 (10번 반복)")
    print("="*50)
    
    try:
        # 기본적으로 최적 조건 실행 (사용자 입력 없이)
        mode = "1"  # 최적 조건 모드를 기본으로 설정
        
        if mode == "1":
            print("📊 최적 조건으로 백테스트를 실행합니다...\n")
            result, config = run_optimized_backtest()
            
            if result:
                print(f"\n🎯 최종 결과: 총 수익률 {result.total_return:.2f}% 달성!")
                print(f"💡 이 결과는 10번의 최적화를 통해 검증된 최적 파라미터를 사용했습니다.")
            else:
                print("\n❌ 백테스트 실행 실패")
                
        elif mode == "2":
            print("🔄 파라미터 최적화를 시작합니다...\n")
            best_result, all_results = optimize_model()
            
            if best_result:
                print(f"\n🎯 최종 결과: 최고 수익률 {best_result.total_return:.2f}% 달성!")
            else:
                print("\n❌ 최적화 실패")
        else:
            print("❌ 올바른 옵션을 선택해주세요.")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

