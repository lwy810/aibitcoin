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

print("🚀 최적화된 스퀴즈 모멘텀 전략")
print("=" * 60)
print("📊 분석 결과를 바탕으로 3가지 최적 전략을 비교합니다")
print("=" * 60)

# ── 3가지 최적화 전략 정의 ─────────────────
strategies = {
    '기본 전략': {
        'position_sizing': 1.0,
        'use_trend_filter': False,
        'use_rsi_filter': False,
        'use_stoploss': False,
        'color': 'red',
        'description': '원래 스퀴즈 모멘텀 전략'
    },
    '균형형 전략': {
        'position_sizing': 0.5,
        'use_trend_filter': True,
        'use_rsi_filter': False,
        'use_stoploss': True,
        'stoploss_pct': 0.05,
        'color': 'blue',
        'description': '수익률과 MDD의 균형을 맞춘 전략'
    },
    '안전형 전략': {
        'position_sizing': 0.3,
        'use_trend_filter': True,
        'use_rsi_filter': True,
        'use_stoploss': False,
        'color': 'green',
        'description': 'MDD 최소화에 최적화된 전략'
    }
}

# ── 백테스트 실행 ─────────────────
def enhanced_backtest(df, val, strategy_name, **kwargs):
    capital = 1.0
    position = 0
    equity = []
    trades = []
    monthly_returns = []
    
    # 파라미터
    position_sizing = kwargs.get('position_sizing', 1.0)
    use_trend_filter = kwargs.get('use_trend_filter', False)
    use_rsi_filter = kwargs.get('use_rsi_filter', False)
    use_stoploss = kwargs.get('use_stoploss', False)
    stoploss_pct = kwargs.get('stoploss_pct', 0.05)
    
    # 시그널 계산
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
            stop_price = entry_price * (1 - stoploss_pct) if use_stoploss else None
        
        # 청산
        elif position > 0:
            should_exit = exit_signal
            if use_stoploss and stop_price and row.close <= stop_price:
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
        
        # 포트폴리오 평가
        if position > 0:
            current_value = capital * (1 + (row.close/entry_price - 1) * position)
        else:
            current_value = capital
        
        equity.append(current_value)
    
    # 성과 지표 계산
    equity_series = pd.Series(equity)
    final_return = equity[-1] - 1
    mdd = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max()
    
    # 샤프 비율 (간단 버전)
    returns = equity_series.pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 6) if returns.std() > 0 else 0  # 4시간 봉
    
    # 거래 통계
    if trades:
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        winning_trades = trades_df['win'].sum()
        win_rate = (winning_trades / total_trades) * 100
        avg_return = trades_df['return_pct'].mean()
        profit_factor = abs(trades_df[trades_df['win']]['return_pct'].sum() / 
                           trades_df[~trades_df['win']]['return_pct'].sum()) if (trades_df['win'] == False).sum() > 0 else float('inf')
        avg_holding_days = trades_df['holding_days'].mean()
    else:
        total_trades = win_rate = avg_return = profit_factor = avg_holding_days = 0
    
    return {
        'strategy': strategy_name,
        'final_return': final_return,
        'mdd': mdd,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'profit_factor': profit_factor,
        'avg_holding_days': avg_holding_days,
        'equity': equity,
        'trades': trades
    }

# 모든 전략 실행
results = {}
for name, params in strategies.items():
    result = enhanced_backtest(df, val, name, **params)
    results[name] = result

# ── 결과 출력 ─────────────────
print(f"\n{'전략명':<15} {'수익률':<10} {'MDD':<8} {'샤프비율':<10} {'거래수':<6} {'승률':<8} {'손익비':<8}")
print("-" * 80)

for name, result in results.items():
    print(f"{result['strategy']:<15} {result['final_return']:>8.1%} {result['mdd']:>6.1%} {result['sharpe_ratio']:>9.2f} {result['total_trades']:>6.0f} {result['win_rate']:>6.1f}% {result['profit_factor']:>7.2f}")

# ── 상세 분석 ─────────────────
print("\n" + "=" * 60)
print("📈 전략별 상세 분석")
print("=" * 60)

for name, result in results.items():
    strategy_info = strategies[name]
    print(f"\n🎯 {name} ({strategy_info['description']})")
    print(f"   💰 최종 수익률: {result['final_return']:.1%}")
    print(f"   📉 최대 낙폭: {result['mdd']:.1%}")
    print(f"   📊 샤프 비율: {result['sharpe_ratio']:.2f}")
    print(f"   🔄 총 거래수: {result['total_trades']:.0f}회")
    print(f"   🎯 승률: {result['win_rate']:.1f}%")
    print(f"   ⚖️ 손익비: {result['profit_factor']:.2f}")
    print(f"   ⏱️ 평균 보유기간: {result['avg_holding_days']:.1f}일")

# ── 위험 조정 수익률 계산 ─────────────────
print("\n" + "=" * 60)
print("⚖️ 위험 조정 성과 비교")
print("=" * 60)

risk_metrics = []
for name, result in results.items():
    risk_return_ratio = result['final_return'] / max(result['mdd'], 0.01)
    calmar_ratio = result['final_return'] / max(result['mdd'], 0.01)  # 연간화 생략
    
    risk_metrics.append({
        'strategy': name,
        'risk_return_ratio': risk_return_ratio,
        'calmar_ratio': calmar_ratio,
        'return_per_trade': result['final_return'] / max(result['total_trades'], 1) * 100
    })

risk_df = pd.DataFrame(risk_metrics).sort_values('risk_return_ratio', ascending=False)

print(f"{'전략명':<15} {'위험수익비':<12} {'거래당수익률':<12}")
print("-" * 45)
for _, row in risk_df.iterrows():
    print(f"{row['strategy']:<15} {row['risk_return_ratio']:>10.2f} {row['return_per_trade']:>10.2f}%")

# ── 시각화 ─────────────────
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

dates = df['timestamp'].values

# 1. 수익 곡선 비교
for name, result in results.items():
    color = strategies[name]['color']
    ax1.plot(dates, result['equity'], label=f"{name} ({result['final_return']:.1%})", 
             color=color, linewidth=2)

ax1.set_title('수익 곡선 비교', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylabel('포트폴리오 가치')

# 2. 드로우다운 비교
for name, result in results.items():
    equity_series = pd.Series(result['equity'])
    drawdown = (equity_series.cummax() - equity_series) / equity_series.cummax() * 100
    color = strategies[name]['color']
    ax2.plot(dates, drawdown, label=f"{name} (MDD: {result['mdd']:.1%})", 
             color=color, linewidth=2)
    ax2.fill_between(dates, 0, drawdown, alpha=0.2, color=color)

ax2.set_title('드로우다운 비교', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylabel('드로우다운 (%)')

# 3. 위험-수익 산점도
returns = [result['final_return']*100 for result in results.values()]
mdds = [result['mdd']*100 for result in results.values()]
colors = [strategies[name]['color'] for name in results.keys()]

scatter = ax3.scatter(mdds, returns, c=colors, s=200, alpha=0.7)
for i, (name, result) in enumerate(results.items()):
    ax3.annotate(name, (result['mdd']*100, result['final_return']*100), 
                xytext=(5, 5), textcoords='offset points', fontweight='bold')

ax3.set_xlabel('최대 낙폭 (%)')
ax3.set_ylabel('수익률 (%)')
ax3.set_title('위험-수익 관계', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. 성과 지표 레이더 차트
metrics = ['수익률', 'MDD역수', '샤프비율', '승률', '손익비']
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # 원형으로 만들기

for name, result in results.items():
    # 정규화 (0-1 스케일)
    values = [
        result['final_return'],  # 수익률
        1 - result['mdd'],       # MDD 역수 (낮을수록 좋음)
        min(result['sharpe_ratio']/20, 1),  # 샤프비율 (20으로 정규화)
        result['win_rate']/100,  # 승률
        min(result['profit_factor']/5, 1)   # 손익비 (5로 정규화)
    ]
    values += values[:1]
    
    color = strategies[name]['color']
    ax4.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
    ax4.fill(angles, values, alpha=0.25, color=color)

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(metrics)
ax4.set_ylim(0, 1)
ax4.set_title('성과 지표 비교', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()

# ── 최종 추천 ─────────────────
print("\n" + "=" * 60)
print("🏆 최종 추천 및 결론")
print("=" * 60)

best_risk_adjusted = max(results.items(), key=lambda x: x[1]['final_return'] / max(x[1]['mdd'], 0.01))
best_return = max(results.items(), key=lambda x: x[1]['final_return'])
best_mdd = min(results.items(), key=lambda x: x[1]['mdd'])

print(f"\n🥇 최고 위험조정수익률: {best_risk_adjusted[0]}")
print(f"🥈 최고 수익률: {best_return[0]}")
print(f"🥉 최저 MDD: {best_mdd[0]}")

print(f"\n💡 투자 성향별 추천:")
print(f"   🔥 공격적 투자자: {best_return[0]} (높은 수익률 추구)")
print(f"   ⚖️ 균형적 투자자: {best_risk_adjusted[0]} (위험 대비 수익 최적화)")
print(f"   🛡️ 보수적 투자자: {best_mdd[0]} (안정성 우선)")

print(f"\n📈 개선 효과:")
baseline = results['기본 전략']
optimized = results['균형형 전략']
print(f"   MDD 개선: {baseline['mdd']:.1%} → {optimized['mdd']:.1%} ({(baseline['mdd']-optimized['mdd'])/baseline['mdd']*100:.1f}% 감소)")
print(f"   샤프비율 개선: {baseline['sharpe_ratio']:.2f} → {optimized['sharpe_ratio']:.2f}")
print(f"   위험수익비: {baseline['final_return']/baseline['mdd']:.2f} → {optimized['final_return']/optimized['mdd']:.2f}")

print(f"\n🎯 핵심 교훈:")
print("1. 포지션 사이징만으로도 MDD를 크게 줄일 수 있음")
print("2. 트렌드 필터는 잘못된 시그널을 효과적으로 걸러냄")
print("3. 과도한 최적화보다는 단순하고 견고한 규칙이 효과적")
print("4. 수익률과 안정성의 트레이드오프를 고려한 균형이 중요")
print("5. 실제 투자 시에는 심리적 요인도 고려해야 함") 