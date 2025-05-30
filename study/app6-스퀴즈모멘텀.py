"# Squeeze Momentum Indicator by LazyBear - Python Implementation with pyupbit" 

import pyupbit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class SqueezeMomentumIndicator:
    def __init__(self, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5, use_true_range=True):
        """
        LazyBear의 Squeeze Momentum Indicator 구현
        
        Parameters:
        bb_length: 볼린저 밴드 길이 (기본값: 20)
        bb_mult: 볼린저 밴드 배수 (기본값: 2.0)
        kc_length: 켈트너 채널 길이 (기본값: 20)
        kc_mult: 켈트너 채널 배수 (기본값: 1.5)
        use_true_range: True Range 사용 여부 (기본값: True)
        """
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        self.use_true_range = use_true_range
    
    def calculate_bollinger_bands(self, close):
        """볼린저 밴드 계산"""
        basis = close.rolling(window=self.bb_length).mean()
        dev = self.bb_mult * close.rolling(window=self.bb_length).std()
        upper_bb = basis + dev
        lower_bb = basis - dev
        return upper_bb, lower_bb, basis
    
    def calculate_true_range(self, high, low, close):
        """True Range 계산"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range
    
    def calculate_keltner_channels(self, high, low, close):
        """켈트너 채널 계산"""
        ma = close.rolling(window=self.kc_length).mean()
        
        if self.use_true_range:
            tr = self.calculate_true_range(high, low, close)
            range_ma = tr.rolling(window=self.kc_length).mean()
        else:
            range_val = high - low
            range_ma = range_val.rolling(window=self.kc_length).mean()
        
        upper_kc = ma + (range_ma * self.kc_mult)
        lower_kc = ma - (range_ma * self.kc_mult)
        
        return upper_kc, lower_kc, ma
    
    def calculate_linear_regression(self, data, length):
        """선형 회귀값 계산 (LazyBear 방식)"""
        if len(data) < length:
            return 0
        
        # 최근 length개 데이터 사용
        y_values = data.values[-length:]
        x_values = np.arange(length)
        
        # 선형 회귀 계산
        n = len(x_values)
        sum_x = np.sum(x_values)
        sum_y = np.sum(y_values)
        sum_xy = np.sum(x_values * y_values)
        sum_x2 = np.sum(x_values ** 2)
        
        # 기울기와 절편 계산
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        # 현재 시점에서의 선형 회귀값 반환
        current_linreg = slope * (length - 1) + intercept
        return current_linreg - y_values[-1]  # LazyBear 방식: 실제값과의 차이
    
    def calculate_momentum(self, high, low, close):
        """모멘텀 계산 (LazyBear 방식)"""
        hl2 = (high + low) / 2
        
        momentum_values = []
        for i in range(len(close)):
            if i < self.kc_length - 1:
                momentum_values.append(0)
            else:
                # 현재까지의 데이터로 계산
                current_close = close.iloc[:i+1]
                current_high = high.iloc[:i+1]
                current_low = low.iloc[:i+1]
                current_hl2 = hl2.iloc[:i+1]
                
                # LazyBear 공식: linreg(source - avg(avg(highest(h, lengthKC), lowest(l, lengthKC)),sma(source,lengthKC)), lengthKC,0)
                highest_h = current_high.rolling(window=self.kc_length).max().iloc[-1]
                lowest_l = current_low.rolling(window=self.kc_length).min().iloc[-1]
                avg1 = (highest_h + lowest_l) / 2
                
                sma_close = current_close.rolling(window=self.kc_length).mean().iloc[-1]
                avg_val = (avg1 + sma_close) / 2
                
                # source - avg_val
                momentum_source = current_close - avg_val
                
                # 선형 회귀 계산
                val = self.calculate_linear_regression(momentum_source.dropna(), self.kc_length)
                momentum_values.append(val)
        
        return pd.Series(momentum_values, index=close.index)
    
    def calculate(self, df):
        """
        Squeeze Momentum Indicator 계산
        
        Parameters:
        df: OHLCV 데이터프레임 (columns: open, high, low, close, volume)
        
        Returns:
        result_df: 계산 결과가 포함된 데이터프레임
        """
        if df.empty or len(df) < max(self.bb_length, self.kc_length):
            raise ValueError("데이터가 부족합니다. 최소한 길이만큼의 데이터가 필요합니다.")
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # 볼린저 밴드 계산
        upper_bb, lower_bb, bb_basis = self.calculate_bollinger_bands(close)
        
        # 켈트너 채널 계산
        upper_kc, lower_kc, kc_ma = self.calculate_keltner_channels(high, low, close)
        
        # Squeeze 조건 확인
        sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
        no_sqz = ~sqz_on & ~sqz_off
        
        # 모멘텀 계산
        momentum = self.calculate_momentum(high, low, close)
        
        # 모멘텀 색상 결정 (LazyBear 방식)
        momentum_color = []
        for i in range(len(momentum)):
            if i == 0:
                momentum_color.append('gray')
            else:
                current_val = momentum.iloc[i]
                prev_val = momentum.iloc[i-1]
                
                if current_val > 0:
                    if current_val > prev_val:
                        momentum_color.append('lime')  # 상승 중인 양수
                    else:
                        momentum_color.append('green')  # 하락 중인 양수
                else:
                    if current_val < prev_val:
                        momentum_color.append('red')  # 하락 중인 음수
                    else:
                        momentum_color.append('maroon')  # 상승 중인 음수
        
        # Squeeze 색상 결정
        squeeze_color = []
        for i in range(len(sqz_on)):
            if no_sqz.iloc[i]:
                squeeze_color.append('blue')
            elif sqz_on.iloc[i]:
                squeeze_color.append('black')
            else:
                squeeze_color.append('gray')
        
        # 결과 데이터프레임 생성
        result_df = df.copy()
        result_df['upper_bb'] = upper_bb
        result_df['lower_bb'] = lower_bb
        result_df['bb_basis'] = bb_basis
        result_df['upper_kc'] = upper_kc
        result_df['lower_kc'] = lower_kc
        result_df['kc_ma'] = kc_ma
        result_df['sqz_on'] = sqz_on
        result_df['sqz_off'] = sqz_off
        result_df['no_sqz'] = no_sqz
        result_df['momentum'] = momentum
        result_df['momentum_color'] = momentum_color
        result_df['squeeze_color'] = squeeze_color
        
        return result_df

def get_upbit_data(ticker, interval="day", count=200):
    """
    업비트에서 데이터 가져오기
    
    Parameters:
    ticker: 티커 (예: "KRW-BTC")
    interval: 시간 간격 ("minute1", "minute3", "minute5", "minute10", "minute15", "minute30", "minute60", "minute240", "day", "week", "month")
    count: 가져올 데이터 개수
    
    Returns:
    df: OHLCV 데이터프레임
    """
    try:
        df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
        if df is None or df.empty:
            raise Exception(f"데이터를 가져올 수 없습니다: {ticker}")
        
        # 컬럼명 소문자로 변경
        df.columns = [col.lower() for col in df.columns]
        df = df.dropna()
        
        return df
    except Exception as e:
        print(f"데이터 가져오기 오류: {e}")
        return None

def plot_squeeze_momentum(df, ticker):
    """
    Squeeze Momentum Indicator 시각화
    
    Parameters:
    df: 계산된 결과 데이터프레임
    ticker: 티커명
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    
    # 가격 차트
    ax1.plot(df.index, df['close'], label='종가', linewidth=2, color='black')
    ax1.plot(df.index, df['upper_bb'], label='볼린저 상단', alpha=0.7, linestyle='--', color='blue')
    ax1.plot(df.index, df['lower_bb'], label='볼린저 하단', alpha=0.7, linestyle='--', color='blue')
    ax1.plot(df.index, df['upper_kc'], label='켈트너 상단', alpha=0.7, linestyle='-.', color='red')
    ax1.plot(df.index, df['lower_kc'], label='켈트나 하단', alpha=0.7, linestyle='-.', color='red')
    ax1.fill_between(df.index, df['upper_bb'], df['lower_bb'], alpha=0.1, color='blue', label='볼린저 밴드')
    ax1.fill_between(df.index, df['upper_kc'], df['lower_kc'], alpha=0.1, color='red', label='켈트너 채널')
    ax1.set_title(f'{ticker} - 가격 차트 (볼린저 밴드 & 켈트너 채널)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # 모멘텀 차트
    # 모멘텀 히스토그램 색상 매핑
    colors = []
    for color in df['momentum_color']:
        if color == 'lime':
            colors.append('lime')
        elif color == 'green':
            colors.append('green')
        elif color == 'red':
            colors.append('red')
        elif color == 'maroon':
            colors.append('maroon')
        else:
            colors.append('gray')
    
    # 모멘텀 히스토그램 그리기
    bars = ax2.bar(df.index, df['momentum'], color=colors, alpha=0.8, width=0.8)
    
    # Squeeze 상태 표시 (0선에 X 마커로)
    for i, (idx, color) in enumerate(zip(df.index, df['squeeze_color'])):
        marker_color = 'black' if color == 'black' else 'gray' if color == 'gray' else 'blue'
        ax2.scatter(idx, 0, color=marker_color, s=80, marker='x', linewidth=3, zorder=5)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax2.set_title(f'{ticker} - Squeeze Momentum Indicator (LazyBear)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('모멘텀', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 범례 추가
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='x', color='w', markerfacecolor='black', markersize=12, linewidth=3, label='스퀴즈 ON (낮은 변동성)'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', markersize=12, linewidth=3, label='스퀴즈 OFF (변동성 확장)'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='blue', markersize=12, linewidth=3, label='스퀴즈 없음'),
        Line2D([0], [0], color='lime', lw=6, label='상승 양수 모멘텀'),
        Line2D([0], [0], color='green', lw=6, label='하락 양수 모멘텀'),
        Line2D([0], [0], color='red', lw=6, label='하락 음수 모멘텀'),
        Line2D([0], [0], color='maroon', lw=6, label='상승 음수 모멘텀'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # X축 날짜 포맷 설정
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df)//10)))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def analyze_signals(df):
    """
    트레이딩 시그널 분석
    
    Parameters:
    df: 계산된 결과 데이터프레임
    
    Returns:
    signals: 시그널 분석 결과
    """
    signals = []
    
    for i in range(1, len(df)):
        current_squeeze = df['squeeze_color'].iloc[i]
        prev_squeeze = df['squeeze_color'].iloc[i-1]
        momentum = df['momentum'].iloc[i]
        momentum_color = df['momentum_color'].iloc[i]
        
        # Squeeze release 감지 (John Carter 방식)
        if prev_squeeze == 'black' and current_squeeze == 'gray':
            if momentum > 0:
                signals.append({
                    'date': df.index[i],
                    'signal': '매수',
                    'reason': '스퀴즈 해제 후 양수 모멘텀',
                    'momentum': momentum,
                    'price': df['close'].iloc[i]
                })
            elif momentum < 0:
                signals.append({
                    'date': df.index[i],
                    'signal': '매도',
                    'reason': '스퀴즈 해제 후 음수 모멘텀',
                    'momentum': momentum,
                    'price': df['close'].iloc[i]
                })
        
        # 모멘텀 방향 변화 감지
        if i > 1:
            prev_momentum_color = df['momentum_color'].iloc[i-1]
            
            # 양수에서 음수로 변화
            if prev_momentum_color in ['lime', 'green'] and momentum_color in ['red', 'maroon']:
                signals.append({
                    'date': df.index[i],
                    'signal': '롱 청산',
                    'reason': '모멘텀이 양수에서 음수로 변화',
                    'momentum': momentum,
                    'price': df['close'].iloc[i]
                })
            
            # 음수에서 양수로 변화
            elif prev_momentum_color in ['red', 'maroon'] and momentum_color in ['lime', 'green']:
                signals.append({
                    'date': df.index[i],
                    'signal': '숏 청산',
                    'reason': '모멘텀이 음수에서 양수로 변화',
                    'momentum': momentum,
                    'price': df['close'].iloc[i]
                })
    
    return signals

def print_current_status(df):
    """현재 상태 출력"""
    latest = df.iloc[-1]
    print("=" * 50)
    print("현재 상태")
    print("=" * 50)
    print(f"날짜: {latest.name.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"종가: {latest['close']:,.0f}원")
    print(f"모멘텀: {latest['momentum']:.6f}")
    print(f"모멘텀 상태: {latest['momentum_color']}")
    
    # 스퀴즈 상태 한글로
    if latest['sqz_on']:
        squeeze_status = "ON (낮은 변동성 - 돌파 대기)"
        status_color = "🖤"
    elif latest['sqz_off']:
        squeeze_status = "OFF (높은 변동성 - 돌파 진행)"
        status_color = "🩶"
    else:
        squeeze_status = "NONE (일반 상태)"
        status_color = "🔵"
    
    print(f"스퀴즈 상태: {status_color} {squeeze_status}")
    
    # 모멘텀 해석
    if latest['momentum'] > 0:
        if latest['momentum_color'] == 'lime':
            momentum_desc = "🟢 강한 상승 모멘텀 (증가 중)"
        else:
            momentum_desc = "🟢 약한 상승 모멘텀 (감소 중)"
    else:
        if latest['momentum_color'] == 'red':
            momentum_desc = "🔴 강한 하락 모멘텀 (증가 중)"
        else:
            momentum_desc = "🔴 약한 하락 모멘텀 (감소 중)"
    
    print(f"모멘텀 해석: {momentum_desc}")
    print()

class SqueezeBacktest:
    """Squeeze Momentum 전략 백테스트"""
    
    def __init__(self, initial_balance=10000000, fee_rate=0.0005):
        """
        초기 설정
        
        Parameters:
        initial_balance: 초기 자본금 (기본값: 1천만원)
        fee_rate: 거래 수수료율 (기본값: 0.05%)
        """
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.reset()
    
    def reset(self):
        """백테스트 상태 초기화"""
        self.balance = self.initial_balance
        self.position = 0  # 보유 수량
        self.position_type = None  # 'long', 'short', None
        self.entry_price = 0
        self.trades = []
        self.portfolio_values = []
        self.daily_returns = []
        
    def calculate_portfolio_value(self, current_price):
        """현재 포트폴리오 가치 계산"""
        if self.position_type == 'long':
            position_value = self.position * current_price
            return self.balance + position_value
        elif self.position_type == 'short':
            # 숏 포지션: 진입가와 현재가의 차이만큼 손익
            pnl = self.position * (self.entry_price - current_price)
            return self.balance + pnl
        else:
            return self.balance
    
    def open_position(self, signal_type, price, date):
        """포지션 오픈"""
        if self.position_type is not None:
            return  # 이미 포지션이 있으면 무시
        
        if signal_type == '매수':
            # 롱 포지션 오픈
            available_balance = self.balance * 0.98  # 수수료 고려해서 98%만 사용
            self.position = available_balance / price
            fee = available_balance * self.fee_rate
            self.balance = self.balance - available_balance - fee
            self.position_type = 'long'
            self.entry_price = price
            
        elif signal_type == '매도':
            # 숏 포지션 오픈 (실제로는 현물에서 불가하지만 백테스트를 위해 가정)
            self.position = self.balance * 0.98 / price  # 숏 수량
            fee = self.balance * 0.98 * self.fee_rate
            self.balance = self.balance - fee
            self.position_type = 'short'
            self.entry_price = price
    
    def close_position(self, price, date, reason):
        """포지션 청산"""
        if self.position_type is None:
            return
        
        if self.position_type == 'long':
            # 롱 포지션 청산
            sell_value = self.position * price
            fee = sell_value * self.fee_rate
            pnl = sell_value - (self.position * self.entry_price) - fee
            
        elif self.position_type == 'short':
            # 숏 포지션 청산
            pnl = self.position * (self.entry_price - price)
            fee = self.position * price * self.fee_rate
            pnl -= fee
        
        # 거래 기록
        trade = {
            'entry_date': getattr(self, 'entry_date', date),
            'exit_date': date,
            'type': self.position_type,
            'entry_price': self.entry_price,
            'exit_price': price,
            'quantity': self.position,
            'pnl': pnl,
            'return_pct': pnl / (self.position * self.entry_price) * 100,
            'reason': reason
        }
        self.trades.append(trade)
        
        # 잔고 업데이트
        self.balance += (self.position * self.entry_price) + pnl
        
        # 포지션 초기화
        self.position = 0
        self.position_type = None
        self.entry_price = 0
    
    def run_backtest(self, df, signals):
        """백테스트 실행"""
        self.reset()
        signal_dict = {signal['date']: signal for signal in signals}
        
        prev_portfolio_value = self.initial_balance
        
        for date, row in df.iterrows():
            current_price = row['close']
            
            # 시그널 처리
            if date in signal_dict:
                signal = signal_dict[date]
                signal_type = signal['signal']
                
                if signal_type in ['매수', '매도']:
                    self.open_position(signal_type, current_price, date)
                    self.entry_date = date
                    
                elif signal_type in ['롱 청산', '숏 청산']:
                    self.close_position(current_price, date, signal['reason'])
            
            # 일일 포트폴리오 가치 계산
            portfolio_value = self.calculate_portfolio_value(current_price)
            self.portfolio_values.append(portfolio_value)
            
            # 일일 수익률 계산
            daily_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
            self.daily_returns.append(daily_return)
            prev_portfolio_value = portfolio_value
        
        # 마지막에 포지션이 남아있으면 청산
        if self.position_type is not None:
            final_price = df['close'].iloc[-1]
            final_date = df.index[-1]
            self.close_position(final_price, final_date, '백테스트 종료')

def calculate_performance_metrics(backtest, df):
    """성과 지표 계산"""
    if len(backtest.portfolio_values) == 0:
        return {}
    
    # 기본 지표
    initial_value = backtest.initial_balance
    final_value = backtest.portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value
    
    # 기간 계산 (일 단위)
    total_days = len(df)
    years = total_days / 365.25
    
    # 연간수익률
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # 일일 수익률 기반 지표
    daily_returns = pd.Series(backtest.daily_returns)
    
    # 변동성 (연환산)
    volatility = daily_returns.std() * np.sqrt(252)
    
    # 샤프 지수 (무위험 수익률 3% 가정)
    risk_free_rate = 0.03
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0
    
    # MDD (Maximum Drawdown) 계산
    portfolio_series = pd.Series(backtest.portfolio_values)
    peak = portfolio_series.expanding().max()
    drawdown = (portfolio_series - peak) / peak
    max_drawdown = drawdown.min()
    
    # 거래 관련 지표
    trades = backtest.trades
    if len(trades) > 0:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trades)
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 최대 연속 승/패
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in trades:
            if trade['pnl'] > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
    
    return {
        'initial_balance': initial_value,
        'final_balance': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'total_days': total_days,
        'years': years
    }

def print_performance_report(metrics, trades):
    """성과 보고서 출력"""
    print("=" * 80)
    print("🏆 백테스트 성과 보고서")
    print("=" * 80)
    
    print("📊 기본 성과 지표")
    print("-" * 50)
    print(f"초기 자본금:     {metrics['initial_balance']:>15,.0f}원")
    print(f"최종 자본금:     {metrics['final_balance']:>15,.0f}원")
    print(f"총 수익률:       {metrics['total_return']:>14.2%}")
    print(f"연간수익률:      {metrics['annual_return']:>14.2%}")
    print(f"변동성(연환산):  {metrics['volatility']:>14.2%}")
    print(f"샤프 지수:       {metrics['sharpe_ratio']:>15.3f}")
    print(f"최대낙폭(MDD):   {metrics['max_drawdown']:>14.2%}")
    print()
    
    print("📈 거래 통계")
    print("-" * 50)
    print(f"총 거래 횟수:    {metrics['total_trades']:>15}회")
    print(f"승률:            {metrics['win_rate']:>14.2%}")
    print(f"평균 승리 손익:  {metrics['avg_win']:>15,.0f}원")
    print(f"평균 손실 손익:  {metrics['avg_loss']:>15,.0f}원")
    print(f"손익비:          {metrics['profit_factor']:>15.3f}")
    print(f"최대 연속 승리:  {metrics['max_consecutive_wins']:>15}회")
    print(f"최대 연속 패배:  {metrics['max_consecutive_losses']:>15}회")
    print()
    
    print(f"📅 백테스트 기간: {metrics['total_days']}일 ({metrics['years']:.2f}년)")
    print()
    
    if len(trades) > 0:
        print("📋 최근 거래 내역 (최대 10개)")
        print("-" * 80)
        print(f"{'날짜':>12} | {'타입':>6} | {'진입가':>10} | {'청산가':>10} | {'수익률':>8} | {'손익':>12}")
        print("-" * 80)
        
        for trade in trades[-10:]:
            entry_date = trade['entry_date'].strftime('%Y-%m-%d')
            trade_type = "롱" if trade['type'] == 'long' else "숏"
            entry_price = f"{trade['entry_price']:,.0f}"
            exit_price = f"{trade['exit_price']:,.0f}"
            return_pct = f"{trade['return_pct']:+.2f}%"
            pnl = f"{trade['pnl']:+,.0f}"
            
            print(f"{entry_date} | {trade_type:>6} | {entry_price:>10} | {exit_price:>10} | {return_pct:>8} | {pnl:>12}")

def plot_backtest_results(df, backtest_result, metrics):
    """백테스트 결과 시각화"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 가격 차트와 거래 포인트
    ax1.plot(df.index, df['close'], label='종가', linewidth=1, color='black', alpha=0.7)
    
    # 거래 포인트 표시
    for trade in backtest_result.trades:
        if trade['type'] == 'long':
            ax1.scatter(trade['entry_date'], trade['entry_price'], color='blue', marker='^', s=100, label='롱 진입' if trade == backtest_result.trades[0] else "")
            ax1.scatter(trade['exit_date'], trade['exit_price'], color='red', marker='v', s=100, label='롱 청산' if trade == backtest_result.trades[0] else "")
        else:
            ax1.scatter(trade['entry_date'], trade['entry_price'], color='orange', marker='v', s=100, label='숏 진입' if not any(t['type'] == 'long' for t in backtest_result.trades) else "")
            ax1.scatter(trade['exit_date'], trade['exit_price'], color='green', marker='^', s=100, label='숏 청산' if not any(t['type'] == 'long' for t in backtest_result.trades) else "")
    
    ax1.set_title('가격 차트 및 거래 포인트', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # 2. 포트폴리오 가치 변화
    portfolio_series = pd.Series(backtest_result.portfolio_values, index=df.index)
    ax2.plot(df.index, portfolio_series, label='포트폴리오 가치', linewidth=2, color='green')
    ax2.axhline(y=backtest_result.initial_balance, color='red', linestyle='--', alpha=0.7, label='초기 자본금')
    ax2.set_title('포트폴리오 가치 변화', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # 3. 드로우다운 차트
    peak = portfolio_series.expanding().max()
    drawdown = (portfolio_series - peak) / peak * 100
    ax3.fill_between(df.index, drawdown, 0, alpha=0.3, color='red', label='드로우다운')
    ax3.axhline(y=metrics['max_drawdown']*100, color='red', linestyle='--', label=f'최대 드로우다운: {metrics["max_drawdown"]*100:.2f}%')
    ax3.set_title('드로우다운 차트', fontsize=14, fontweight='bold')
    ax3.set_ylabel('드로우다운 (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 월별 수익률 히트맵 (간단 버전)
    if len(backtest_result.daily_returns) > 30:  # 최소 30일 이상의 데이터가 있을 때
        daily_ret_series = pd.Series(backtest_result.daily_returns, index=df.index)
        monthly_returns = daily_ret_series.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        months = [date.strftime('%Y-%m') for date in monthly_returns.index]
        returns = monthly_returns.values
        
        colors = ['red' if r < 0 else 'green' for r in returns]
        bars = ax4.bar(range(len(returns)), returns, color=colors, alpha=0.7)
        ax4.set_title('월별 수익률', fontsize=14, fontweight='bold')
        ax4.set_ylabel('수익률 (%)')
        ax4.set_xticks(range(len(months)))
        ax4.set_xticklabels(months, rotation=45)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        # 수익률 값 표시
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1.5),
                    f'{ret:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    else:
        ax4.text(0.5, 0.5, '월별 데이터 부족\n(최소 30일 필요)', transform=ax4.transAxes, 
                ha='center', va='center', fontsize=12)
        ax4.set_title('월별 수익률', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# 메인 실행 부분
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Squeeze Momentum Indicator by LazyBear")
    print("💰 업비트 pyupbit 구현 버전")
    print("=" * 60)
    
    # 설정
    ticker = "KRW-BTC"  # 비트코인 (다른 코인도 가능: KRW-ETH, KRW-ADA 등)
    interval = "day"    # 일봉 (minute1, minute5, minute15, minute30, minute60, minute240, day, week, month)
    count = 365         # 데이터 개수
    
    print(f"📊 분석 설정")
    print(f"   - 티커: {ticker}")
    print(f"   - 시간간격: {interval}")
    print(f"   - 데이터 개수: {count}개")
    print()
    
    # 데이터 가져오기
    print("📡 업비트에서 데이터를 가져오는 중...")
    df = get_upbit_data(ticker, interval, count)
    
    if df is None:
        print("❌ 데이터를 가져올 수 없습니다.")
        exit()
    
    print(f"✅ 데이터 수집 완료: {len(df)}개")
    print(f"📅 기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print()
    
    # Squeeze Momentum Indicator 계산
    print("🔧 Squeeze Momentum Indicator 계산 중...")
    smi = SqueezeMomentumIndicator(
        bb_length=20,        # 볼린저 밴드 길이
        bb_mult=2.0,         # 볼린저 밴드 표준편차 배수
        kc_length=20,        # 켈트너 채널 길이
        kc_mult=1.5,         # 켈트너 채널 ATR 배수
        use_true_range=True  # True Range 사용
    )
    
    try:
        result_df = smi.calculate(df)
        print("✅ 계산 완료!")
        print()
        
        # 현재 상태 출력
        print_current_status(result_df)
        
        # 시그널 분석
        print("🔍 트레이딩 시그널 분석")
        print("=" * 50)
        signals = analyze_signals(result_df)
        
        if signals:
            print(f"📈 최근 시그널 (최대 10개):")
            for signal in signals[-10:]:
                date_str = signal['date'].strftime('%Y-%m-%d')
                signal_emoji = "🟢" if signal['signal'] in ['매수', '숏 청산'] else "🔴" if signal['signal'] in ['매도', '롱 청산'] else "⚪"
                print(f"   {date_str} | {signal_emoji} {signal['signal']:>6} | {signal['reason']} | 가격: {signal['price']:,.0f}원")
        else:
            print("📭 최근 시그널이 없습니다.")
        
        print()
        print("📊 최근 20일간 상세 데이터")
        print("=" * 50)
        recent_data = result_df[['close', 'momentum', 'momentum_color', 'squeeze_color', 'sqz_on', 'sqz_off']].tail(20)
        
        # 데이터 포맷팅해서 출력
        for idx, row in recent_data.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            momentum_emoji = "🟢" if row['momentum'] > 0 else "🔴"
            squeeze_emoji = "🖤" if row['squeeze_color'] == 'black' else "🩶" if row['squeeze_color'] == 'gray' else "🔵"
            print(f"{date_str} | 가격: {row['close']:>8,.0f} | {momentum_emoji} 모멘텀: {row['momentum']:>8.4f} | {squeeze_emoji} {row['squeeze_color']}")
        
        print()
        print("📈 차트를 생성하는 중...")
        plot_squeeze_momentum(result_df, ticker)
        
        print("�� 분석 완료!")
        
        # 백테스트 실행
        print("🔧 백테스트 실행 중...")
        backtest = SqueezeBacktest(initial_balance=10000000, fee_rate=0.0005)
        backtest.run_backtest(df, signals)
        print("✅ 백테스트 완료!")
        
        # 성과 지표 계산
        metrics = calculate_performance_metrics(backtest, df)
        print("✅ 성과 지표 계산 완료!")
        
        # 성과 보고서 출력
        print_performance_report(metrics, backtest.trades)
        
        # 백테스트 결과 시각화
        plot_backtest_results(df, backtest, metrics)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}") 
