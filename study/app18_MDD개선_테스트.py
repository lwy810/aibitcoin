import pandas as pd, numpy as np

df = pd.read_csv('btc_4h_data_2018_to_2025.csv', parse_dates=['timestamp'])
df = df[(df.timestamp >= '2019-01-01') & (df.timestamp <= '2025-06-03')]

# ── 기술적 지표 계산 ─────────────────
def calculate_squeeze_momentum(df, length=20, mult=2.0, lengthKC=20, multKC=1.5):
    src = df['close']
    basis = src.rolling(length).mean()
    dev = mult * src.rolling(length).std()
    upperBB, lowerBB = basis + dev, basis - dev
    
    ma = src.rolling(lengthKC).mean()
    rng = pd.concat([df['high']-df['low'],
                     (df['high']-df['close'].shift()).abs(),
                     (df['low']-df['close'].shift()).abs()],
                    axis=1).max(axis=1)
    rngMA = rng.rolling(lengthKC).mean()
    upperKC, lowerKC = ma + rngMA*multKC, ma - rngMA*multKC
    
    sqz_on = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqz_off = (lowerBB < lowerKC) & (upperBB > upperKC)
    
    highest = df['high'].rolling(lengthKC).max()
    lowest = df['low'].rolling(lengthKC).min()
    val = (src - ((highest+lowest)/2 + ma)/2)\
            .rolling(lengthKC).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
    
    return val, sqz_on, sqz_off

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 기본 지표 계산
val, sqz_on, sqz_off = calculate_squeeze_momentum(df)
df['rsi'] = calculate_rsi(df['close'])
df['sma_20'] = df['close'].rolling(20).mean()
df['sma_50'] = df['close'].rolling(50).mean()
df['atr'] = pd.concat([df['high']-df['low'],
                       (df['high']-df['close'].shift()).abs(),
                       (df['low']-df['close'].shift()).abs()],
                      axis=1).max(axis=1).rolling(14).mean()

# ── 백테스트 함수 ─────────────────
def backtest_strategy(df, val, strategy_name, **kwargs):
    capital = 1.0
    position = 0
    equity = []
    trades = []
    
    # 전략별 파라미터
    use_stoploss = kwargs.get('use_stoploss', False)
    stoploss_pct = kwargs.get('stoploss_pct', 0.05)
    use_trend_filter = kwargs.get('use_trend_filter', False)
    use_rsi_filter = kwargs.get('use_rsi_filter', False)
    position_sizing = kwargs.get('position_sizing', 1.0)  # 고정 비율 투자
    use_atr_stoploss = kwargs.get('use_atr_stoploss', False)
    atr_multiplier = kwargs.get('atr_multiplier', 2.0)
    
    # 시그널 미리 계산
    entry_signals = (val.shift(1) < 0) & (val > 0)
    exit_signals = (val.shift(1) > 0) & (val < 0)
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # 기본 진입/청산 시그널
        entry_signal = entry_signals.iloc[i] if i < len(entry_signals) else False
        exit_signal = exit_signals.iloc[i] if i < len(exit_signals) else False
        
        # 필터 적용
        if use_trend_filter and not pd.isna(row.sma_20) and not pd.isna(row.sma_50):
            entry_signal = entry_signal and (row.close > row.sma_20 > row.sma_50)
        
        if use_rsi_filter and not pd.isna(row.rsi):
            entry_signal = entry_signal and (30 < row.rsi < 70)
        
        # 진입
        if entry_signal and position == 0:
            entry_price = row.close
            entry_time = row.timestamp
            position = position_sizing  # 포지션 사이즈 조정
            
            # 스톱로스 설정
            if use_stoploss:
                stop_price = entry_price * (1 - stoploss_pct)
            elif use_atr_stoploss and not pd.isna(row.atr):
                stop_price = entry_price - (row.atr * atr_multiplier)
            else:
                stop_price = None
        
        # 청산 (시그널 또는 스톱로스)
        elif position > 0:
            should_exit = exit_signal
            
            # 스톱로스 체크
            if stop_price and row.close <= stop_price:
                should_exit = True
            
            if should_exit:
                exit_price = row.close
                exit_time = row.timestamp
                trade_return = (exit_price / entry_price - 1) * position
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': trade_return * 100,
                    'holding_days': (exit_time - entry_time).days,
                    'win': trade_return > 0,
                    'position_size': position
                })
                
                capital *= (1 + trade_return)
                position = 0
                stop_price = None
        
        # 포지션이 있으면 현재 가격으로 평가, 없으면 현금
        if position > 0:
            equity.append(capital * (1 + (row.close/entry_price - 1) * position))
        else:
            equity.append(capital)
    
    # 결과 계산
    if len(equity) > 0:
        final_return = equity[-1] - 1
        mdd = ((pd.Series(equity).cummax() - pd.Series(equity)) / pd.Series(equity).cummax()).max()
    else:
        final_return = 0
        mdd = 0
    
    # 거래 통계
    if trades:
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        winning_trades = trades_df['win'].sum()
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        avg_return = trades_df['return_pct'].mean()
        profit_factor = abs(trades_df[trades_df['win']]['return_pct'].sum() / 
                           trades_df[~trades_df['win']]['return_pct'].sum()) if (trades_df['win'] == False).sum() > 0 else float('inf')
    else:
        total_trades = win_rate = avg_return = profit_factor = 0
    
    return {
        'strategy': strategy_name,
        'final_return': final_return,
        'mdd': mdd,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'profit_factor': profit_factor
    }

# ── 다양한 전략 테스트 ─────────────────
print("🔬 MDD 개선 전략 테스트 시작...")
print("=" * 80)

strategies = [
    # 기본 전략
    {'name': '1️⃣ 기본 전략', 'params': {}},
    
    # 스톱로스 전략들
    {'name': '2️⃣ 스톱로스 3%', 'params': {'use_stoploss': True, 'stoploss_pct': 0.03}},
    {'name': '3️⃣ 스톱로스 5%', 'params': {'use_stoploss': True, 'stoploss_pct': 0.05}},
    {'name': '4️⃣ 스톱로스 8%', 'params': {'use_stoploss': True, 'stoploss_pct': 0.08}},
    
    # ATR 기반 스톱로스
    {'name': '5️⃣ ATR 스톱로스 1.5x', 'params': {'use_atr_stoploss': True, 'atr_multiplier': 1.5}},
    {'name': '6️⃣ ATR 스톱로스 2.0x', 'params': {'use_atr_stoploss': True, 'atr_multiplier': 2.0}},
    
    # 포지션 사이징
    {'name': '7️⃣ 50% 포지션', 'params': {'position_sizing': 0.5}},
    {'name': '8️⃣ 70% 포지션', 'params': {'position_sizing': 0.7}},
    
    # 트렌드 필터
    {'name': '9️⃣ 트렌드 필터', 'params': {'use_trend_filter': True}},
    {'name': '🔟 트렌드 + 5% 스톱', 'params': {'use_trend_filter': True, 'use_stoploss': True, 'stoploss_pct': 0.05}},
    
    # RSI 필터
    {'name': '1️⃣1️⃣ RSI 필터', 'params': {'use_rsi_filter': True}},
    {'name': '1️⃣2️⃣ RSI + 5% 스톱', 'params': {'use_rsi_filter': True, 'use_stoploss': True, 'stoploss_pct': 0.05}},
    
    # 복합 전략
    {'name': '1️⃣3️⃣ 트렌드+RSI+70%포지션', 'params': {'use_trend_filter': True, 'use_rsi_filter': True, 'position_sizing': 0.7}},
    {'name': '1️⃣4️⃣ 복합: 트렌드+RSI+ATR스톱', 'params': {'use_trend_filter': True, 'use_rsi_filter': True, 'use_atr_stoploss': True, 'atr_multiplier': 2.0}},
    {'name': '1️⃣5️⃣ 보수적: 50%포지션+5%스톱+트렌드', 'params': {'position_sizing': 0.5, 'use_stoploss': True, 'stoploss_pct': 0.05, 'use_trend_filter': True}}
]

results = []
for strategy in strategies:
    result = backtest_strategy(df, val, strategy['name'], **strategy['params'])
    results.append(result)

# ── 결과 출력 ─────────────────
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mdd')  # MDD 기준 정렬

print(f"{'전략명':<35} {'수익률':<10} {'MDD':<8} {'거래수':<6} {'승률':<8} {'수익비':<8}")
print("-" * 80)

for _, row in results_df.iterrows():
    print(f"{row['strategy']:<35} {row['final_return']:>8.1%} {row['mdd']:>6.1%} {row['total_trades']:>6.0f} {row['win_rate']:>6.1f}% {row['profit_factor']:>7.2f}")

print("\n" + "=" * 80)
print("🏆 MDD 개선 우수 전략 TOP 3:")
print("=" * 80)

top3 = results_df.head(3)
for i, (_, row) in enumerate(top3.iterrows(), 1):
    print(f"{i}위: {row['strategy']}")
    print(f"    💰 수익률: {row['final_return']:.1%}")
    print(f"    📉 MDD: {row['mdd']:.1%}")
    print(f"    📊 거래수: {row['total_trades']:.0f}회")
    print(f"    🎯 승률: {row['win_rate']:.1f}%")
    print(f"    ⚖️ 손익비: {row['profit_factor']:.2f}")
    print()

print("💡 분석 결과:")
best_strategy = results_df.iloc[0]
original_mdd = results_df[results_df['strategy'] == '1️⃣ 기본 전략']['mdd'].iloc[0]
mdd_improvement = original_mdd - best_strategy['mdd']

print(f"   📈 MDD 개선: {original_mdd:.1%} → {best_strategy['mdd']:.1%} ({mdd_improvement:.1%}p 개선)")
print(f"   🔥 최고 성과: {best_strategy['strategy']}") 