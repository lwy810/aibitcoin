import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import ta
import warnings
import time
warnings.filterwarnings('ignore')

class BitcoinAITradingStrategy:
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
        # 초기 가중치 설정
        self.weights = {
            'rsi': 0.2,
            'macd': 0.4, 
            'google_trends': 0.2,
            'ml_signal': 0.4
        }
        
        # 거래 임계값
        self.buy_threshold = 0.5
        self.sell_threshold = -0.5
        
    def get_bitcoin_data(self, start_date='2018-01-01', end_date='2024-01-31'):
        """비트코인 가격 데이터 수집"""
        btc = yf.Ticker("BTC-USD")
        data = btc.history(start=start_date, end=end_date)
        return data
    
    def calculate_rsi(self, data, period=14):
        """RSI 계산 (14일 기간)"""
        rsi = ta.momentum.RSIIndicator(close=data['Close'], window=period)
        return rsi.rsi()
    
    def calculate_macd(self, data):
        """MACD 계산"""
        macd = ta.trend.MACD(close=data['Close'])
        return macd.macd(), macd.macd_signal(), macd.macd_diff()
    
    def get_google_trends(self, start_date='2018-01-01', end_date='2024-01-31'):
        """Google 트렌드 데이터 수집"""
        try:
            # 특정 기간의 비트코인 검색 트렌드
            time.sleep(3)  # 요청 간격 늘리기
            
            # 연도별로 나누어서 요청 (Google Trends 제한 우회)
            all_trends = pd.DataFrame()
            
            start_year = 2018
            end_year = 2024
            
            for year in range(start_year, end_year + 1):
                try:
                    timeframe = f'{year}-01-01 {year}-12-31'
                    if year == end_year:
                        timeframe = f'{year}-01-01 {end_date}'
                    
                    print(f"   Google Trends 데이터 수집 중... {year}년")
                    self.pytrends.build_payload(['bitcoin'], timeframe=timeframe, geo='')
                    time.sleep(3)
                    
                    yearly_trends = self.pytrends.interest_over_time()
                    if not yearly_trends.empty:
                        yearly_trends = yearly_trends.drop(columns=['isPartial'], errors='ignore')
                        yearly_trends.columns = ['bitcoin_trend']
                        all_trends = pd.concat([all_trends, yearly_trends])
                    
                    time.sleep(2)  # 추가 대기
                    
                except Exception as year_error:
                    print(f"   {year}년 Google Trends 데이터 수집 실패: {year_error}")
                    continue
            
            return all_trends
            
        except Exception as e:
            print(f"Google Trends 데이터 수집 오류: {e}")
            print("Google Trends 없이 계속 진행합니다...")
            return pd.DataFrame()
    
    def generate_rsi_signal(self, rsi):
        """RSI 신호 생성"""
        signals = []
        for value in rsi:
            if pd.isna(value):
                signals.append(0)
            elif value > 70:  # 과매수
                signals.append(-1)
            elif value < 30:  # 과매도
                signals.append(1)
            else:
                signals.append(0)
        return np.array(signals)
    
    def generate_macd_signal(self, macd_line, macd_signal):
        """MACD 신호 생성"""
        signals = []
        for i in range(len(macd_line)):
            if pd.isna(macd_line.iloc[i]) or pd.isna(macd_signal.iloc[i]):
                signals.append(0)
            elif macd_line.iloc[i] > macd_signal.iloc[i]:  # 골든 크로스
                signals.append(1)
            else:  # 데드 크로스
                signals.append(-1)
        return np.array(signals)
    
    def generate_google_trends_signal(self, trends_data):
        """Google 트렌드 신호 생성"""
        if trends_data.empty:
            return np.array([0] * len(trends_data))
        
        # 7일 이동평균 계산
        trends_data['ma7'] = trends_data['bitcoin_trend'].rolling(window=7).mean()
        
        signals = []
        for i in range(len(trends_data)):
            current = trends_data['bitcoin_trend'].iloc[i]
            ma7 = trends_data['ma7'].iloc[i]
            
            if pd.isna(current) or pd.isna(ma7):
                signals.append(0)
            elif current > ma7:  # 현재 관심도가 7일 평균 초과
                signals.append(1)
            else:
                signals.append(-1)
                
        return np.array(signals)
    
    def prepare_ml_features(self, data, rsi, macd_line, macd_signal, trends_signals):
        """머신러닝 모델용 특성 준비"""
        features = pd.DataFrame()
        
        # 기술적 지표
        features['rsi'] = rsi
        features['macd'] = macd_line
        features['macd_signal'] = macd_signal
        features['macd_diff'] = macd_line - macd_signal
        
        # 가격 기반 특성
        features['price_change'] = data['Close'].pct_change()
        features['volume_change'] = data['Volume'].pct_change()
        features['high_low_ratio'] = data['High'] / data['Low']
        
        # 이동평균
        features['sma_20'] = data['Close'].rolling(window=20).mean()
        features['sma_50'] = data['Close'].rolling(window=50).mean()
        features['price_sma20_ratio'] = data['Close'] / features['sma_20']
        
        # Google 트렌드 신호 (길이 맞추기)
        if len(trends_signals) < len(features):
            trends_extended = np.pad(trends_signals, (len(features) - len(trends_signals), 0), 'constant')
        else:
            trends_extended = trends_signals[:len(features)]
        
        features['google_trends_signal'] = trends_extended
        
        # NaN 값 처리
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def train_ml_model(self, features, target):
        """랜덤 포레스트 모델 훈련"""
        # 타겟 생성: 다음 날 가격 상승 여부
        X = features[:-1]  # 마지막 행 제외 (타겟이 없으므로)
        y = target[1:]     # 첫 번째 행 제외 (이전 날 데이터 없으므로)
        
        # 훈련/테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 모델 훈련
        self.rf_model.fit(X_train, y_train)
        
        # 성능 평가
        y_pred = self.rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"랜덤 포레스트 모델 정확도: {accuracy:.4f}")
        print(f"분류 보고서:\n{classification_report(y_test, y_pred)}")
        
        self.is_trained = True
        return accuracy
    
    def generate_ml_signals(self, features):
        """머신러닝 모델을 사용한 신호 생성"""
        if not self.is_trained:
            print("경고: 모델이 훈련되지 않았습니다.")
            return np.zeros(len(features))
        
        predictions = self.rf_model.predict(features)
        # 1은 매수 신호, 0은 매도 신호로 변환
        ml_signals = np.where(predictions == 1, 1, -1)
        return ml_signals
    
    def calculate_weighted_score(self, rsi_signals, macd_signals, trends_signals, ml_signals):
        """가중치 점수 계산"""
        # 모든 신호의 길이를 맞춤
        min_length = min(len(rsi_signals), len(macd_signals), len(trends_signals), len(ml_signals))
        
        rsi_signals = rsi_signals[:min_length]
        macd_signals = macd_signals[:min_length]
        trends_signals = trends_signals[:min_length]
        ml_signals = ml_signals[:min_length]
        
        weighted_scores = (
            self.weights['rsi'] * rsi_signals +
            self.weights['macd'] * macd_signals +
            self.weights['google_trends'] * trends_signals +
            self.weights['ml_signal'] * ml_signals
        )
        
        return weighted_scores
    
    def generate_trading_decisions(self, weighted_scores):
        """거래 결정 생성"""
        decisions = []
        for score in weighted_scores:
            if score > self.buy_threshold:
                decisions.append('BUY')
            elif score < self.sell_threshold:
                decisions.append('SELL')
            else:
                decisions.append('HOLD')
        
        return decisions
    
    def backtest_strategy(self, data, decisions):
        """백테스팅 수행"""
        if len(decisions) != len(data):
            min_length = min(len(decisions), len(data))
            decisions = decisions[:min_length]
            data = data.iloc[:min_length]
        
        portfolio_value = 10000  # 초기 자본
        btc_holdings = 0
        cash = portfolio_value
        portfolio_values = []
        
        for i in range(len(decisions)):
            current_price = data['Close'].iloc[i]
            decision = decisions[i]
            
            if decision == 'BUY' and cash > current_price:
                # 매수
                btc_to_buy = cash / current_price
                btc_holdings += btc_to_buy
                cash = 0
                
            elif decision == 'SELL' and btc_holdings > 0:
                # 매도
                cash = btc_holdings * current_price
                btc_holdings = 0
            
            # 포트폴리오 가치 계산
            portfolio_value = cash + (btc_holdings * current_price)
            portfolio_values.append(portfolio_value)
        
        return portfolio_values
    
    def run_strategy(self):
        """전체 전략 실행"""
        print("=== AI 기반 앙상블 비트코인 트레이딩 전략 (2018-2024) ===")
        
        # 1. 데이터 수집
        print("1. 비트코인 데이터 수집 중... (2018년 1월 ~ 2024년 1월)")
        btc_data = self.get_bitcoin_data(start_date='2018-01-01', end_date='2024-01-31')
        print(f"   수집된 데이터: {len(btc_data)}일 ({btc_data.index[0].strftime('%Y-%m-%d')} ~ {btc_data.index[-1].strftime('%Y-%m-%d')})")
        
        print("2. Google 트렌드 데이터 수집 중...")
        trends_data = self.get_google_trends(start_date='2018-01-01', end_date='2024-01-31')
        if not trends_data.empty:
            print(f"   Google Trends 데이터: {len(trends_data)}일")
        else:
            print("   Google Trends 데이터 수집 실패 - 다른 지표로 계속 진행")
        
        # 2. 기술적 지표 계산
        print("3. 기술적 지표 계산 중...")
        rsi = self.calculate_rsi(btc_data)
        macd_line, macd_signal, macd_diff = self.calculate_macd(btc_data)
        
        # 3. 신호 생성
        print("4. 신호 생성 중...")
        rsi_signals = self.generate_rsi_signal(rsi)
        macd_signals = self.generate_macd_signal(macd_line, macd_signal)
        
        if not trends_data.empty:
            trends_signals = self.generate_google_trends_signal(trends_data)
        else:
            trends_signals = np.zeros(len(btc_data))
        
        # 4. 머신러닝 모델 훈련
        print("5. 머신러닝 모델 훈련 중...")
        ml_features = self.prepare_ml_features(btc_data, rsi, macd_line, macd_signal, trends_signals)
        
        # 타겟 변수: 다음 날 가격 상승 여부
        target = (btc_data['Close'].shift(-1) > btc_data['Close']).astype(int)
        
        accuracy = self.train_ml_model(ml_features, target)
        
        # 5. ML 신호 생성
        ml_signals = self.generate_ml_signals(ml_features)
        
        # 6. 가중치 점수 계산
        print("6. 가중치 점수 계산 중...")
        weighted_scores = self.calculate_weighted_score(rsi_signals, macd_signals, trends_signals, ml_signals)
        
        # 7. 거래 결정
        print("7. 거래 결정 생성 중...")
        trading_decisions = self.generate_trading_decisions(weighted_scores)
        
        # 8. 백테스팅
        print("8. 백테스팅 수행 중...")
        portfolio_values = self.backtest_strategy(btc_data, trading_decisions)
        
        # 9. 결과 분석
        print("9. 결과 분석...")
        initial_value = 10000
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        # 최대 낙폭 계산
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak * 100
        max_drawdown = np.min(drawdown)
        
        # 승률 계산
        buy_signals = [i for i, decision in enumerate(trading_decisions) if decision == 'BUY']
        winning_trades = 0
        total_trades = len(buy_signals) - 1
        
        for i in range(len(buy_signals) - 1):
            buy_index = buy_signals[i]
            sell_index = buy_signals[i + 1] if i + 1 < len(buy_signals) else len(btc_data) - 1
            
            if sell_index < len(btc_data):
                buy_price = btc_data['Close'].iloc[buy_index]
                sell_price = btc_data['Close'].iloc[sell_index]
                if sell_price > buy_price:
                    winning_trades += 1
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 결과 출력
        print("\n=== 백테스트 결과 (2018년 1월 ~ 2024년 1월) ===")
        print(f"분석 기간: {btc_data.index[0].strftime('%Y-%m-%d')} ~ {btc_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"총 분석 일수: {len(btc_data)}일")
        print(f"총 수익률: {total_return:.2f}%")
        print(f"연환산 수익률: {(total_return / len(btc_data) * 365):.2f}%")
        print(f"최대 낙폭 (MDD): {max_drawdown:.2f}%")
        print(f"승률: {win_rate:.2f}%")
        print(f"총 거래 횟수: {total_trades}")
        print(f"ML 모델 정확도: {accuracy:.2f}%")
        print(f"초기 자본: ${initial_value:,.2f}")
        print(f"최종 자본: ${final_value:,.2f}")
        
        # 비트코인 홀드 대비 성과
        btc_initial = btc_data['Close'].iloc[0]
        btc_final = btc_data['Close'].iloc[-1]
        btc_return = (btc_final - btc_initial) / btc_initial * 100
        outperformance = total_return - btc_return
        
        print(f"\n=== 벤치마크 비교 ===")
        print(f"비트코인 홀드 수익률: {btc_return:.2f}%")
        print(f"전략 대비 홀드 성과: {outperformance:+.2f}%p")
        if outperformance > 0:
            print("🎉 전략이 단순 홀드보다 우수한 성과를 보였습니다!")
        else:
            print("📉 전략이 단순 홀드보다 저조한 성과를 보였습니다.")
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'ml_accuracy': accuracy,
            'portfolio_values': portfolio_values,
            'trading_decisions': trading_decisions,
            'btc_data': btc_data
        }

# 실행 예제
if __name__ == "__main__":
    strategy = BitcoinAITradingStrategy()
    results = strategy.run_strategy()
    
    # 추가 분석을 위한 데이터프레임 생성
    analysis_df = pd.DataFrame({
        'Date': results['btc_data'].index[:len(results['trading_decisions'])],
        'Price': results['btc_data']['Close'][:len(results['trading_decisions'])],
        'Decision': results['trading_decisions'],
        'Portfolio_Value': results['portfolio_values']
    })
    
    print("\n=== 최근 10개 거래 신호 ===")
    print(analysis_df.tail(10))