import pandas as pd, numpy as np

df = pd.read_csv('btc_4h_data_2018_to_2025.csv', parse_dates=['timestamp'])
df = df[(df.timestamp >= '2019-01-01') & (df.timestamp <= '2025-06-03')]

# ── 지표 파라미터 ─────────────────
length, mult = 20, 2.0
lengthKC, multKC = 20, 1.5
use_tr = True

# ── Squeeze Momentum 계산 ─────────
src = df['close']; basis = src.rolling(length).mean()
dev = mult * src.rolling(length).std()
upperBB, lowerBB = basis + dev, basis - dev

ma = src.rolling(lengthKC).mean()
rng = (df['high'] - df['low']) if not use_tr else \
      pd.concat([df['high']-df['low'],
                 (df['high']-df['close'].shift()).abs(),
                 (df['low']-df['close'].shift()).abs()],
                axis=1).max(axis=1)
rngMA = rng.rolling(lengthKC).mean()
upperKC, lowerKC = ma + rngMA*multKC, ma - rngMA*multKC

sqz_on  = (lowerBB > lowerKC) & (upperBB < upperKC)
sqz_off = (lowerBB < lowerKC) & (upperBB > upperKC)

highest = df['high'].rolling(lengthKC).max()
lowest  = df['low'].rolling(lengthKC).min()
val = (src - ((highest+lowest)/2 + ma)/2)\
        .rolling(lengthKC).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)

# ── 진입‧청산 ──────────────────────
entry = (val.shift(1) < 0) & (val > 0)
exit  = (val.shift(1) > 0) & (val < 0)

capital = 1.0
position, equity = 0, []
trades = []  # 거래 내역 저장

for idx, row in df.iterrows():
    if entry[idx] and position == 0:
        entry_price, position = row.close, 1
        entry_time = row.timestamp
    elif exit[idx] and position == 1:
        exit_price = row.close
        exit_time = row.timestamp
        trade_return = exit_price / entry_price - 1
        trade_profit = trade_return * 100  # 퍼센트로 변환
        
        # 거래 정보 저장
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return_pct': trade_profit,
            'holding_days': (exit_time - entry_time).days,
            'win': trade_return > 0
        })
        
        capital *= exit_price / entry_price
        position = 0
    equity.append(capital * (row.close/entry_price if position else 1))

df['equity'] = equity
mdd = ((df['equity'].cummax() - df['equity']) / df['equity'].cummax()).max()

# ── 거래 통계 계산 ─────────────────
if trades:
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    winning_trades = trades_df['win'].sum()
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades) * 100
    
    avg_return = trades_df['return_pct'].mean()
    avg_winning_return = trades_df[trades_df['win']]['return_pct'].mean() if winning_trades > 0 else 0
    avg_losing_return = trades_df[~trades_df['win']]['return_pct'].mean() if losing_trades > 0 else 0
    
    best_trade = trades_df['return_pct'].max()
    worst_trade = trades_df['return_pct'].min()
    avg_holding_days = trades_df['holding_days'].mean()
    
    profit_factor = abs(trades_df[trades_df['win']]['return_pct'].sum() / 
                       trades_df[~trades_df['win']]['return_pct'].sum()) if losing_trades > 0 else float('inf')

print("=" * 50)
print("📊 스퀴즈 모멘텀 백테스트 결과 (2019-2025)")
print("=" * 50)
print(f"💰 총 수익률: {equity[-1]-1:.2%}")
print(f"💸 수익금 (초기 100): {(equity[-1]-1)*100:.2f}")
print(f"📉 최대 낙폭 (MDD): {mdd:.2%}")

if trades:
    print(f"\n📈 거래 통계:")
    print(f"   총 거래 횟수: {total_trades:,}회")
    print(f"   승리 거래: {winning_trades:,}회")
    print(f"   패배 거래: {losing_trades:,}회")
    print(f"   🎯 승률: {win_rate:.2f}%")
    print(f"   📊 평균 수익률: {avg_return:.2f}%")
    print(f"   🟢 평균 승리 수익률: {avg_winning_return:.2f}%")
    print(f"   🔴 평균 손실 수익률: {avg_losing_return:.2f}%")
    print(f"   🚀 최고 수익 거래: {best_trade:.2f}%")
    print(f"   💥 최악 손실 거래: {worst_trade:.2f}%")
    print(f"   ⏱️  평균 보유 기간: {avg_holding_days:.1f}일")
    print(f"   ⚖️  손익비 (Profit Factor): {profit_factor:.2f}")
else:
    print(f"\n⚠️  거래 내역이 없습니다.")

print("=" * 50)
