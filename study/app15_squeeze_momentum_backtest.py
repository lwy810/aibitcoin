import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SqueezeIndicator:
    def __init__(self, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5, use_true_range=True):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        self.use_true_range = use_true_range
    
    def calculate_true_range(self, df):
        """True Range 계산"""
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
        low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
        
        tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        return tr
    
    def calculate_bollinger_bands(self, df):
        """볼린저 밴드 계산"""
        source = df['Close']
        basis = source.rolling(window=self.bb_length).mean()
        dev = self.bb_mult * source.rolling(window=self.bb_length).std()
        
        upper_bb = basis + dev
        lower_bb = basis - dev
        
        return upper_bb, lower_bb, basis
    
    def calculate_keltner_channels(self, df):
        """켈트너 채널 계산"""
        source = df['Close']
        ma = source.rolling(window=self.kc_length).mean()
        
        if self.use_true_range:
            range_data = self.calculate_true_range(df)
        else:
            range_data = df['High'] - df['Low']
        
        range_ma = range_data.rolling(window=self.kc_length).mean()
        
        upper_kc = ma + range_ma * self.kc_mult
        lower_kc = ma - range_ma * self.kc_mult
        
        return upper_kc, lower_kc, ma
    
    def calculate_linear_regression(self, series, length):
        """선형 회귀 계산"""
        def linreg(y):
            if len(y) < length:
                return np.nan
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            return slope * (len(y) - 1) + intercept  # 마지막 값 예측
        
        return series.rolling(window=length).apply(linreg, raw=False)
    
    def calculate_momentum_value(self, df):
        """모멘텀 값 계산"""
        source = df['Close']
        
        # 최고가와 최저가의 평균
        highest_high = df['High'].rolling(window=self.kc_length).max()
        lowest_low = df['Low'].rolling(window=self.kc_length).min()
        hl_avg = (highest_high + lowest_low) / 2
        
        # 종가의 이동평균
        close_ma = source.rolling(window=self.kc_length).mean()
        
        # 두 값의 평균
        avg_val = (hl_avg + close_ma) / 2
        
        # 종가에서 평균값을 뺀 후 선형 회귀
        diff = source - avg_val
        momentum = self.calculate_linear_regression(diff, self.kc_length)
        
        return momentum
    
    def calculate_indicators(self, df):
        """모든 지표 계산"""
        # 볼린저 밴드
        upper_bb, lower_bb, bb_basis = self.calculate_bollinger_bands(df)
        
        # 켈트너 채널
        upper_kc, lower_kc, kc_ma = self.calculate_keltner_channels(df)
        
        # Squeeze 상태 판단
        sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)  # Squeeze 발생
        sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)  # Squeeze 해제
        no_sqz = ~sqz_on & ~sqz_off  # 중간 상태
        
        # 모멘텀 값
        momentum = self.calculate_momentum_value(df)
        
        # 결과를 DataFrame에 추가
        result_df = df.copy()
        result_df['Upper_BB'] = upper_bb
        result_df['Lower_BB'] = lower_bb
        result_df['BB_Basis'] = bb_basis
        result_df['Upper_KC'] = upper_kc
        result_df['Lower_KC'] = lower_kc
        result_df['KC_MA'] = kc_ma
        result_df['Squeeze_On'] = sqz_on
        result_df['Squeeze_Off'] = sqz_off
        result_df['No_Squeeze'] = no_sqz
        result_df['Momentum'] = momentum
        result_df['Momentum_Prev'] = momentum.shift(1)
        result_df['Momentum_Color'] = np.where(momentum > 0,
                                               np.where(momentum > momentum.shift(1), 'lime', 'green'),
                                               np.where(momentum < momentum.shift(1), 'red', 'maroon'))
        
        return result_df

def load_bitcoin_data():
    """비트코인 데이터 로딩 (2018-01-01 ~ 2023-12-31)"""
    try:
        df = pd.read_csv('../bitcoin_processed_data.csv', index_col='Date', parse_dates=True)
        
        # 2018-01-01부터 2023-12-31까지 데이터 선택
        start_date = '2018-01-01'
        end_date = '2023-12-31'
        
        df_period = df[(df.index >= start_date) & (df.index <= end_date)].copy()
        
        print(f"데이터 기간: {df_period.index.min().strftime('%Y-%m-%d')} ~ {df_period.index.max().strftime('%Y-%m-%d')}")
        print(f"총 {len(df_period)}일 데이터 (약 {len(df_period)/365:.1f}년)")
        
        return df_period
        
    except FileNotFoundError:
        print("bitcoin_processed_data.csv 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"데이터 로딩 오류: {e}")
        return None

def backtest_squeeze_strategy(df, initial_capital=10000, transaction_cost=0.0005):
    """Squeeze Momentum 전략 백테스트"""
    print(f"\n=== Squeeze Momentum 백테스트 시작 ===")
    print(f"초기 자본: ${initial_capital:,.2f}")
    print(f"거래 수수료: {transaction_cost*100:.2f}%")
    
    capital = initial_capital
    holdings = 0
    entry_price = 0
    trades = []
    portfolio_values = []
    
    for i, (date, row) in enumerate(df.iterrows()):
        current_price = row['Close']
        momentum = row['Momentum']
        momentum_prev = row['Momentum_Prev']
        squeeze_on = row['Squeeze_On']
        squeeze_off = row['Squeeze_Off']
        
        # 포트폴리오 가치 계산
        portfolio_value = capital + holdings * current_price
        portfolio_values.append(portfolio_value)
        
        # 거래 신호 생성
        signal = 0
        
        # 조건 1: 모멘텀이 음수에서 양수로 전환 (매수)
        if (momentum > 0 and momentum_prev <= 0) and not pd.isna(momentum) and not pd.isna(momentum_prev):
            signal = 1
            
        # 조건 2: 모멘텀이 양수에서 음수로 전환 (매도)
        elif (momentum < 0 and momentum_prev >= 0) and not pd.isna(momentum) and not pd.isna(momentum_prev):
            signal = -1
            
        # 조건 3: Squeeze 해제 시 모멘텀 방향으로 거래
        elif squeeze_off and not pd.isna(momentum):
            if momentum > 0:
                signal = 1
            elif momentum < 0:
                signal = -1
        
        # 거래 실행
        if signal == 1 and capital > 0 and holdings == 0:  # 매수
            invest_amount = capital * 0.95  # 95% 투자
            cost = invest_amount * (1 + transaction_cost)
            if cost <= capital:
                holdings = invest_amount / current_price
                capital -= cost
                entry_price = current_price
                
                trades.append({
                    'Date': date,
                    'Action': 'BUY',
                    'Price': current_price,
                    'Amount': holdings,
                    'Capital': capital,
                    'Momentum': momentum,
                    'Squeeze_Status': 'On' if squeeze_on else ('Off' if squeeze_off else 'None')
                })
                
        elif signal == -1 and holdings > 0:  # 매도
            proceeds = holdings * current_price * (1 - transaction_cost)
            capital += proceeds
            
            profit = proceeds - (holdings * entry_price * (1 + transaction_cost))
            
            trades.append({
                'Date': date,
                'Action': 'SELL',
                'Price': current_price,
                'Amount': holdings,
                'Capital': capital,
                'Profit': profit,
                'Momentum': momentum,
                'Squeeze_Status': 'On' if squeeze_on else ('Off' if squeeze_off else 'None')
            })
            
            holdings = 0
            entry_price = 0
    
    # 마지막에 보유 중이면 청산
    if holdings > 0:
        final_price = df['Close'].iloc[-1]
        final_proceeds = holdings * final_price * (1 - transaction_cost)
        capital += final_proceeds
        
        profit = final_proceeds - (holdings * entry_price * (1 + transaction_cost))
        
        trades.append({
            'Date': df.index[-1],
            'Action': 'FINAL_SELL',
            'Price': final_price,
            'Amount': holdings,
            'Capital': capital,
            'Profit': profit,
            'Momentum': df['Momentum'].iloc[-1],
            'Squeeze_Status': 'FINAL'
        })
        
        holdings = 0
    
    # 성과 분석
    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    # MDD 계산
    portfolio_series = pd.Series(portfolio_values, index=df.index)
    rolling_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series / rolling_max) - 1
    max_drawdown = drawdown.min() * 100
    
    # 거래 분석
    buy_trades = [t for t in trades if t['Action'] == 'BUY']
    sell_trades = [t for t in trades if t['Action'] in ['SELL', 'FINAL_SELL']]
    
    profitable_trades = len([t for t in sell_trades if t.get('Profit', 0) > 0])
    total_trades = len(sell_trades)
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # 결과 출력
    print(f"\n=== 백테스트 결과 ===")
    print(f"최종 자본: ${final_capital:,.2f}")
    print(f"총 수익률: {total_return:.2f}%")
    
    # 연평균 수익률 계산
    years = len(df) / 365.25  # 실제 투자 기간 (년)
    annualized_return = (final_capital / initial_capital) ** (1/years) - 1
    print(f"연평균 수익률 (CAGR): {annualized_return*100:.2f}%")
    print(f"투자 기간: {years:.1f}년")
    
    print(f"최대 낙폭 (MDD): {max_drawdown:.2f}%")
    print(f"총 거래 횟수: {len(trades)}")
    print(f"매수 거래: {len(buy_trades)}회")
    print(f"매도 거래: {len(sell_trades)}회")
    print(f"승률: {win_rate:.2f}%")
    
    if sell_trades:
        profits = [t.get('Profit', 0) for t in sell_trades]
        avg_profit = np.mean(profits)
        print(f"평균 거래 손익: ${avg_profit:.2f}")
        
        # 샤프 비율 계산 (단순화)
        if len(profits) > 1:
            profit_std = np.std(profits)
            if profit_std > 0:
                sharpe_ratio = avg_profit / profit_std
                print(f"샤프 비율 (단순화): {sharpe_ratio:.2f}")
    
    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'annualized_return': annualized_return * 100,
        'years': years,
        'max_drawdown': max_drawdown,
        'trades': trades,
        'portfolio_values': portfolio_series,
        'win_rate': win_rate
    }

def plot_results(df, results):
    """결과 시각화"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 가격과 볼린저 밴드, 켈트너 채널
    ax1.plot(df.index, df['Close'], label='Bitcoin Price', color='black', linewidth=1)
    ax1.plot(df.index, df['Upper_BB'], label='Upper BB', color='blue', alpha=0.7)
    ax1.plot(df.index, df['Lower_BB'], label='Lower BB', color='blue', alpha=0.7)
    ax1.plot(df.index, df['Upper_KC'], label='Upper KC', color='red', alpha=0.7)
    ax1.plot(df.index, df['Lower_KC'], label='Lower KC', color='red', alpha=0.7)
    ax1.fill_between(df.index, df['Upper_BB'], df['Lower_BB'], alpha=0.1, color='blue')
    ax1.fill_between(df.index, df['Upper_KC'], df['Lower_KC'], alpha=0.1, color='red')
    
    # 거래 표시
    trades = results['trades']
    buy_trades = [t for t in trades if t['Action'] == 'BUY']
    sell_trades = [t for t in trades if t['Action'] in ['SELL', 'FINAL_SELL']]
    
    if buy_trades:
        buy_dates = [t['Date'] for t in buy_trades]
        buy_prices = [t['Price'] for t in buy_trades]
        ax1.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy', zorder=5)
    
    if sell_trades:
        sell_dates = [t['Date'] for t in sell_trades]
        sell_prices = [t['Price'] for t in sell_trades]
        ax1.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell', zorder=5)
    
    ax1.set_title('Bitcoin Price with Bollinger Bands & Keltner Channels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Squeeze 상태
    squeeze_colors = []
    for _, row in df.iterrows():
        if row['Squeeze_On']:
            squeeze_colors.append('black')
        elif row['Squeeze_Off']:
            squeeze_colors.append('gray')
        else:
            squeeze_colors.append('blue')
    
    ax2.scatter(df.index, [0] * len(df), c=squeeze_colors, s=20, alpha=0.7)
    ax2.set_title('Squeeze Status (Black=On, Gray=Off, Blue=None)')
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(True, alpha=0.3)
    
    # 3. 모멘텀 히스토그램
    momentum_colors = df['Momentum'].apply(lambda x: 'green' if x > 0 else 'red')
    ax3.bar(df.index, df['Momentum'], color=momentum_colors, alpha=0.7, width=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Momentum Histogram')
    ax3.grid(True, alpha=0.3)
    
    # 4. 포트폴리오 가치
    ax4.plot(df.index, results['portfolio_values'], color='purple', linewidth=2, label='Portfolio Value')
    ax4.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax4.set_title(f"Portfolio Value (Return: {results['total_return']:.2f}%)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    print("=== Squeeze Momentum Indicator 백테스트 ===\n")
    
    # 데이터 로딩
    df = load_bitcoin_data()
    if df is None:
        return
    
    # Squeeze Indicator 계산
    squeeze = SqueezeIndicator(
        bb_length=20,
        bb_mult=2.0,
        kc_length=20,
        kc_mult=1.5,
        use_true_range=True
    )
    
    df_with_indicators = squeeze.calculate_indicators(df)
    
    # 처음 몇 개 행은 지표 계산이 불완전하므로 제거
    df_clean = df_with_indicators.dropna().copy()
    
    print(f"지표 계산 후 유효 데이터: {len(df_clean)}일")
    
    # 백테스트 실행
    results = backtest_squeeze_strategy(df_clean)
    
    # 결과 시각화는 생략 (matplotlib 충돌 방지)
    # plot_results(df_clean, results)
    
    # 거래 내역 출력
    print(f"\n=== 거래 내역 ===")
    for i, trade in enumerate(results['trades']):
        action = trade['Action']
        date = trade['Date'].strftime('%Y-%m-%d')
        price = trade['Price']
        momentum = trade['Momentum']
        squeeze_status = trade['Squeeze_Status']
        
        if action == 'BUY':
            print(f"{i+1:2d}. {date} | {action:4s} | ${price:8.2f} | Momentum: {momentum:6.2f} | Squeeze: {squeeze_status}")
        else:
            profit = trade.get('Profit', 0)
            print(f"{i+1:2d}. {date} | {action:4s} | ${price:8.2f} | Momentum: {momentum:6.2f} | Squeeze: {squeeze_status} | Profit: ${profit:.2f}")
    
    # 추가 통계
    print(f"\n=== 상세 분석 ===")
    buy_trades = [t for t in results['trades'] if t['Action'] == 'BUY']
    sell_trades = [t for t in results['trades'] if t['Action'] in ['SELL', 'FINAL_SELL']]
    
    if sell_trades:
        profits = [t.get('Profit', 0) for t in sell_trades]
        positive_profits = [p for p in profits if p > 0]
        negative_profits = [p for p in profits if p < 0]
        
        print(f"수익 거래: {len(positive_profits)}회, 평균 수익: ${np.mean(positive_profits):.2f}" if positive_profits else "수익 거래: 0회")
        print(f"손실 거래: {len(negative_profits)}회, 평균 손실: ${np.mean(negative_profits):.2f}" if negative_profits else "손실 거래: 0회")
        print(f"최대 수익: ${max(profits):.2f}")
        print(f"최대 손실: ${min(profits):.2f}")
        
        # Squeeze 상태별 거래 분석
        squeeze_on_trades = [t for t in buy_trades if t['Squeeze_Status'] == 'On']
        squeeze_off_trades = [t for t in buy_trades if t['Squeeze_Status'] == 'Off']
        squeeze_none_trades = [t for t in buy_trades if t['Squeeze_Status'] == 'None']
        
        print(f"\nSqueeze 상태별 매수:")
        print(f"  Squeeze On: {len(squeeze_on_trades)}회")
        print(f"  Squeeze Off: {len(squeeze_off_trades)}회") 
        print(f"  Squeeze None: {len(squeeze_none_trades)}회")
        
        # 연도별 성과 분석
        print(f"\n=== 연도별 성과 ===")
        portfolio_values = results['portfolio_values']
        yearly_returns = {}
        
        for year in range(2018, 2024):
            year_start = f"{year}-01-01"
            year_end = f"{year}-12-31"
            
            year_data = portfolio_values[(portfolio_values.index >= year_start) & 
                                        (portfolio_values.index <= year_end)]
            
            if len(year_data) > 0:
                if year == 2018:
                    start_value = 10000  # 초기 자본
                else:
                    prev_year_end = f"{year-1}-12-31"
                    prev_data = portfolio_values[portfolio_values.index <= prev_year_end]
                    start_value = prev_data.iloc[-1] if len(prev_data) > 0 else 10000
                
                end_value = year_data.iloc[-1]
                yearly_return = (end_value / start_value - 1) * 100
                yearly_returns[year] = yearly_return
                
                print(f"  {year}년: {yearly_return:6.2f}% (${start_value:8,.0f} → ${end_value:8,.0f})")
        
        if yearly_returns:
            avg_yearly_return = np.mean(list(yearly_returns.values()))
            print(f"  평균: {avg_yearly_return:6.2f}%")

if __name__ == "__main__":
    main() 