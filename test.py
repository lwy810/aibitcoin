import pybithumb
import time
from datetime import datetime
from dotenv import load_dotenv
import os

# .env 파일에서 빗썸 API 키 로드 (없으면 주석 처리)
load_dotenv()
API_KEY = os.environ.get("BITHUMB_ACCESS_KEY")
SECRET_KEY = os.environ.get("BITHUMB_SECRET_KEY")

def print_with_time(message):
    """타임스탬프와 함께 메시지 출력"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")

def run_basic_examples():
    """기본적인 빗썸 API 활용 예제"""
    print_with_time("=== 빗썸 API 기본 예제 ===")
    
    # 1. 모든 티커 목록 조회
    tickers = pybithumb.get_tickers()
    print_with_time(f"총 {len(tickers)}개의 코인이 거래중입니다.")
    print_with_time(f"첫 10개 코인: {tickers[:10]}")
    
    # 2. 특정 코인(BTC) 현재가 조회
    coin_symbol = "BTC"  # 비트코인
    current_price = pybithumb.get_current_price(coin_symbol)
    print_with_time(f"{coin_symbol} 현재가: {current_price:,}원")
    
    # 3. 여러 코인 현재가 한번에 조회
    coins_to_check = ["BTC", "ETH", "XRP", "DOGE"]
    multiple_prices = pybithumb.get_current_price(coins_to_check)
    
    for coin in coins_to_check:
        if coin in multiple_prices:
            price = multiple_prices[coin]['closing_price']
            print_with_time(f"{coin} 현재가: {float(price):,}원")
    
    # 4. 24시간 거래 정보 조회 (시가/고가/저가/종가/거래량)
    btc_ohlcv = pybithumb.get_market_detail(coin_symbol)
    print_with_time(f"{coin_symbol} OHLCV: {btc_ohlcv}")
    
    # 시가/고가/저가/종가/거래량을 개별적으로 출력
    if btc_ohlcv:
        open_price, high_price, low_price, close_price, volume = btc_ohlcv
        print_with_time(f"{coin_symbol} 시가: {open_price:,}원")
        print_with_time(f"{coin_symbol} 고가: {high_price:,}원")
        print_with_time(f"{coin_symbol} 저가: {low_price:,}원")
        print_with_time(f"{coin_symbol} 종가: {close_price:,}원")
        print_with_time(f"{coin_symbol} 거래량: {volume:,}")

def run_private_api_examples():
    """인증이 필요한 빗썸 API 활용 예제"""
    if not API_KEY or not SECRET_KEY:
        print_with_time("API 키가 설정되지 않아 인증 API 예제를 실행할 수 없습니다.")
        print_with_time(".env 파일에 BITHUMB_ACCESS_KEY와 BITHUMB_SECRET_KEY를 설정하세요.")
        return
    
    print_with_time("\n=== 빗썸 인증 API 예제 ===")
    
    try:
        # 빗썸 객체 생성
        bithumb = pybithumb.Bithumb(API_KEY, SECRET_KEY)
        
        # 1. 잔고 조회
        # 반환값: (총 잔고, 거래중 잔고, 사용 가능 잔고, 평균 매수가)
        balance = bithumb.get_balance("BTC")
        if balance:
            print_with_time(f"BTC 총 보유량: {balance[0]}")
            print_with_time(f"BTC 거래중 수량: {balance[1]}")
            print_with_time(f"BTC 사용가능 수량: {balance[2]}")
            print_with_time(f"BTC 평균 매수가: {balance[3]:,}원")
        
        # 2. 원화 잔고 조회
        krw_balance = bithumb.get_balance("KRW")
        if krw_balance:
            print_with_time(f"원화 총액: {krw_balance[0]:,}원")
            print_with_time(f"원화 거래중: {krw_balance[1]:,}원")
            print_with_time(f"원화 사용가능: {krw_balance[2]:,}원")
        
        # 3. 미체결 주문 조회 (예시)
        # 참고: 실제 미체결 주문이 없으면 빈 리스트 반환
        orders = bithumb.get_outstanding_order("BTC")
        if orders:
            print_with_time(f"미체결 주문: {orders}")
        else:
            print_with_time("BTC 미체결 주문이 없습니다.")
            
    except Exception as e:
        print_with_time(f"인증 API 사용 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    # 공개 API 예제 실행 (인증 불필요)
    run_basic_examples()
    
    # 잠시 대기
    time.sleep(1)
    
    # 인증 API 예제 실행 (API 키 필요)
    run_private_api_examples()
