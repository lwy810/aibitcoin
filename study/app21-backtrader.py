import backtrader as bt
import datetime
import yfinance as yf
import pandas as pd

# 1. 전략 생성 (SMA Crossover Strategy)
class SmaCross(bt.Strategy):
    # 사용할 파라미터 정의 (단기 SMA 기간, 장기 SMA 기간)
    params = (
        ('pfast', 10),  # 단기 SMA 기간
        ('pslow', 30),  # 장기 SMA 기간
    )

    def __init__(self):
        # 사용할 지표들을 미리 계산
        self.dataclose = self.datas[0].close  # 종가 데이터

        # 이동 평균선 계산
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.pfast)
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.pslow)

        # 이동 평균선 교차 신호 계산
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        # 매수/매도 조건 확인
        if not self.position:  # 현재 포지션이 없으면
            if self.crossover > 0:  # 단기 SMA가 장기 SMA를 상향 돌파 (골든 크로스)
                self.buy()  # 매수
        elif self.crossover < 0:  # 단기 SMA가 장기 SMA를 하향 돌파 (데드 크로스)
            self.close()  # 현재 포지션 청산 (매도)

    def log(self, txt, dt=None):
        ''' 로깅 함수 '''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 주문 접수/승인 상태 - 특별한 동작 없음
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'매수 실행, 가격: {order.executed.price:.2f}, 비용: {order.executed.value:.2f}, 수수료: {order.executed.comm:.2f}'
                )
            elif order.issell():
                self.log(
                    f'매도 실행, 가격: {order.executed.price:.2f}, 비용: {order.executed.value:.2f}, 수수료: {order.executed.comm:.2f}'
                )
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('주문 취소/마진콜/거절')

        self.order = None # 다음 주문을 위해 주문 상태 초기화

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'거래 이익, 총이익 {trade.pnl:.2f}, 순이익 {trade.pnlcomm:.2f}')


if __name__ == '__main__':
    # 2. Cerebro 엔진 초기화
    cerebro = bt.Cerebro()

    # 3. 실제 데이터 다운로드 및 추가
    print("데이터 다운로드 중...")
    
    # 다양한 데이터 옵션 (원하는 것을 선택하세요)
    # SYMBOL_OPTIONS = {
    #     "애플": "AAPL",
    #     "테슬라": "TSLA", 
    #     "비트코인": "BTC-USD",
    #     "이더리움": "ETH-USD",
    #     "S&P500": "^GSPC",
    #     "나스닥": "^IXIC"
    # }
    
    # 다양한 투자 상품 선택 (원하는 것으로 변경하세요)
    SYMBOLS = {
        "애플": "AAPL",
        "마이크로소프트": "MSFT",
        "테슬라": "TSLA", 
        "구글": "GOOGL",
        "아마존": "AMZN",
        "비트코인": "BTC-USD",
        "이더리움": "ETH-USD",
        "S&P500": "^GSPC",
        "나스닥": "^IXIC",
        "원유": "CL=F",
        "금": "GC=F"
    }
    
    # 여기서 원하는 종목을 선택하세요
    symbol = SYMBOLS["비트코인"]  # 비트코인으로 변경 (다른 종목으로 바꿀 수 있음)
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    try:
        # Yahoo Finance에서 실제 데이터 다운로드
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"❌ {symbol} 데이터를 가져올 수 없습니다.")
            raise ValueError(f"데이터를 가져올 수 없습니다: {symbol}")
        
        print(f"✅ {symbol} 데이터 다운로드 완료")
        print(f"📅 기간: {start_date} ~ {end_date}")
        print(f"📊 데이터 수: {len(df)}일")
        print(f"💰 시작가: ${df['Close'].iloc[0]:.2f}")
        print(f"💰 종료가: ${df['Close'].iloc[-1]:.2f}")
        
        # 컬럼명을 백트레이더 형식에 맞게 변경
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # 필요한 컬럼만 선택
        df['OpenInterest'] = 0  # OpenInterest 컬럼 추가 (기본값 0)
        
        # PandasData로 데이터 생성
        data = bt.feeds.PandasData(
            dataname=df,
            fromdate=datetime.datetime.strptime(start_date, "%Y-%m-%d"),
            todate=datetime.datetime.strptime(end_date, "%Y-%m-%d")
        )
        
    except Exception as e:
        print(f"❌ 데이터 다운로드 실패: {e}")
        print("기본 가상 데이터를 사용합니다...")
        
        # 오류 시 기본 가상 데이터 생성
        import numpy as np
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        price = 100.0
        prices = []
        
        for _ in range(len(dates)):
            change = np.random.normal(0, 2)
            price += change
            prices.append(max(price, 10))
        
        data_dict = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            volatility = abs(np.random.normal(0, 1))
            open_price = prices[i-1] if i > 0 else close
            data_dict.append({
                'Open': open_price,
                'High': close + volatility,
                'Low': close - volatility,
                'Close': close,
                'Volume': np.random.randint(1000, 10000),
                'OpenInterest': 0
            })
        
        df = pd.DataFrame(data_dict, index=dates)
        data = bt.feeds.PandasData(dataname=df)


    cerebro.adddata(data)

    # 4. 전략 추가
    cerebro.addstrategy(SmaCross)

    # 5. 초기 자본금 설정
    cerebro.broker.setcash(100000.0)

    # 6. 매매 단위 설정 (예: 한 번에 10주씩 거래)
    cerebro.addsizer(bt.sizers.SizerFix, stake=10)

    # 7. 수수료 설정 (예: 0.1%)
    cerebro.broker.setcommission(commission=0.001)

    # 8. 백테스팅 실행 전 초기 포트폴리오 가치 출력
    print(f'시작 포트폴리오 가치: {cerebro.broker.getvalue():.2f}')

    # 9. 백테스팅 실행
    results = cerebro.run()

    # 10. 최종 포트폴리오 가치 출력
    print(f'최종 포트폴리오 가치: {cerebro.broker.getvalue():.2f}')

    # 11. (선택 사항) 결과 플롯
    # matplotlib 라이브러리 필요: pip install matplotlib
    try:
        cerebro.plot()
    except Exception as e:
        print(f"플롯 생성 중 오류 발생 (matplotlib 라이브러리가 설치되어 있는지 확인하세요): {e}")