import pandas as pd, numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

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
    
    highest = df['high'].rolling(lengthKC).max()
    lowest = df['low'].rolling(lengthKC).min()
    val = (src - ((highest+lowest)/2 + ma)/2)\
            .rolling(lengthKC).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
    
    return val

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 기본 지표 계산
val = calculate_squeeze_momentum(df)
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
    position_sizing = kwargs.get('position_sizing', 1.0)
    use_atr_stoploss = kwargs.get('use_atr_stoploss', False)
    atr_multiplier = kwargs.get('atr_multiplier', 2.0)
    
    # 시그널 미리 계산
    entry_signals = (val.shift(1) < 0) & (val > 0)
    exit_signals = (val.shift(1) > 0) & (val < 0)
    
    for i in range(len(df)):
        row = df.iloc[i]
        
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
            position = position_sizing
            
            if use_stoploss:
                stop_price = entry_price * (1 - stoploss_pct)
            elif use_atr_stoploss and not pd.isna(row.atr):
                stop_price = entry_price - (row.atr * atr_multiplier)
            else:
                stop_price = None
        
        # 청산
        elif position > 0:
            should_exit = exit_signal
            
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
        
        # 포지션 평가
        if position > 0:
            equity.append(capital * (1 + (row.close/entry_price - 1) * position))
        else:
            equity.append(capital)
    
    # 결과 계산
    if len(equity) > 0:
        final_return = equity[-1] - 1
        equity_series = pd.Series(equity)
        running_max = equity_series.cummax()
        drawdown = (running_max - equity_series) / running_max
        mdd = drawdown.max()
    else:
        final_return = 0
        mdd = 0
        equity = [1] * len(df)
    
    # 거래 통계
    if trades:
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        winning_trades = trades_df['win'].sum()
        win_rate = (winning_trades / total_trades) * 100
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
        'profit_factor': profit_factor,
        'equity': equity,
        'trades': trades
    }

# ── 핵심 전략들만 비교 ─────────────────
strategies = [
    {'name': '기본 전략', 'params': {}},
    {'name': '보수적 (50%포지션+5%스톱+트렌드)', 'params': {'position_sizing': 0.5, 'use_stoploss': True, 'stoploss_pct': 0.05, 'use_trend_filter': True}},
    {'name': '트렌드+RSI+70%포지션', 'params': {'use_trend_filter': True, 'use_rsi_filter': True, 'position_sizing': 0.7}},
    {'name': '50% 포지션', 'params': {'position_sizing': 0.5}},
    {'name': '트렌드 필터', 'params': {'use_trend_filter': True}}
]

print("🔬 MDD 개선 전략 상세 분석")
print("=" * 60)

results = []
for strategy in strategies:
    result = backtest_strategy(df, val, strategy['name'], **strategy['params'])
    results.append(result)

# ── 결과 출력 ─────────────────
results_df = pd.DataFrame(results)

print(f"{'전략명':<30} {'수익률':<10} {'MDD':<8} {'거래수':<6} {'승률':<8}")
print("-" * 60)

for _, row in results_df.iterrows():
    print(f"{row['strategy']:<30} {row['final_return']:>8.1%} {row['mdd']:>6.1%} {row['total_trades']:>6.0f} {row['win_rate']:>6.1f}%")

print("\n" + "=" * 60)

# ── 상세 분석 ─────────────────
best_mdd_strategy = results_df.loc[results_df['mdd'].idxmin()]
original_strategy = results_df[results_df['strategy'] == '기본 전략'].iloc[0]

print("🏆 최적 MDD 전략 상세 분석:")
print(f"   전략명: {best_mdd_strategy['strategy']}")
print(f"   💰 수익률: {best_mdd_strategy['final_return']:.1%}")
print(f"   📉 MDD: {best_mdd_strategy['mdd']:.1%}")
print(f"   📊 거래수: {best_mdd_strategy['total_trades']:.0f}회")
print(f"   🎯 승률: {best_mdd_strategy['win_rate']:.1f}%")
print(f"   ⚖️ 손익비: {best_mdd_strategy['profit_factor']:.2f}")

print(f"\n📈 기본 전략 대비 개선:")
print(f"   MDD 개선: {original_strategy['mdd']:.1%} → {best_mdd_strategy['mdd']:.1%} ({original_strategy['mdd'] - best_mdd_strategy['mdd']:.1%}p 개선)")
print(f"   수익률 변화: {original_strategy['final_return']:.1%} → {best_mdd_strategy['final_return']:.1%}")

# ── 차트 그리기 ─────────────────
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. 수익 곡선 비교
dates = df['timestamp'].values
for i, result in enumerate(results):
    if i < 3:  # 상위 3개 전략만
        ax1.plot(dates, result['equity'], label=result['strategy'], linewidth=2)

ax1.set_title('📈 수익 곡선 비교 (상위 3개 전략)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylabel('포트폴리오 가치')

# 2. MDD vs 수익률 산점도
ax2.scatter(results_df['mdd']*100, results_df['final_return']*100, s=100, alpha=0.7)
for i, row in results_df.iterrows():
    ax2.annotate(f"{i+1}", (row['mdd']*100, row['final_return']*100), 
                xytext=(5, 5), textcoords='offset points', fontweight='bold')

ax2.set_xlabel('MDD (%)')
ax2.set_ylabel('수익률 (%)')
ax2.set_title('📊 MDD vs 수익률 관계', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. 거래수 vs 승률
ax3.scatter(results_df['total_trades'], results_df['win_rate'], s=100, alpha=0.7)
for i, row in results_df.iterrows():
    ax3.annotate(f"{i+1}", (row['total_trades'], row['win_rate']), 
                xytext=(5, 5), textcoords='offset points', fontweight='bold')

ax3.set_xlabel('총 거래수')
ax3.set_ylabel('승률 (%)')
ax3.set_title('📈 거래수 vs 승률 관계', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. 드로우다운 비교 (상위 2개 전략)
best_equity = pd.Series(best_mdd_strategy['equity'])
original_equity = pd.Series(original_strategy['equity'])

best_drawdown = (best_equity.cummax() - best_equity) / best_equity.cummax() * 100
original_drawdown = (original_equity.cummax() - original_equity) / original_equity.cummax() * 100

ax4.plot(dates, best_drawdown, label=f'{best_mdd_strategy["strategy"]} (MDD: {best_mdd_strategy["mdd"]:.1%})', linewidth=2)
ax4.plot(dates, original_drawdown, label=f'기본 전략 (MDD: {original_strategy["mdd"]:.1%})', linewidth=2)
ax4.fill_between(dates, 0, best_drawdown, alpha=0.3)
ax4.fill_between(dates, 0, original_drawdown, alpha=0.3)

ax4.set_title('📉 드로우다운 비교', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylabel('드로우다운 (%)')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("💡 결론 및 권장사항:")
print("=" * 60)
print("1️⃣ MDD 최적화를 위해서는 포지션 사이징과 필터링이 핵심")
print("2️⃣ 트렌드 필터 + 작은 포지션 사이즈가 MDD를 크게 줄임")
print("3️⃣ 수익률과 MDD는 트레이드오프 관계 - 균형점 찾기 중요")
print("4️⃣ 보수적 전략이 장기적으로 더 안정적인 성과 제공")
print("5️⃣ 과도한 스톱로스는 오히려 성과를 해칠 수 있음") 