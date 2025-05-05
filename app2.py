import os
import time
import uuid
from dotenv import load_dotenv
import pyupbit

# .env 파일에서 환경변수 로드
load_dotenv()

# 업비트 API 키 설정
access_key = os.environ.get("UPBIT_ACCESS_KEY")
secret_key = os.environ.get("UPBIT_SECRET_KEY")

# 업비트 객체 생성
upbit = pyupbit.Upbit(access_key, secret_key)

# USD 코인 (KRW-USDT) 실제 매수/매도 함수
def trading_real():
    # 타겟 코인 설정
    coin = "KRW-USDT"
    
    # 현재 가격 확인
    current_price = pyupbit.get_current_price(coin)
    print(f"현재 가격: {current_price}원")
    
    # 매수 가격 설정
    buy_price = current_price
    
    # 목표 판매 가격 설정 (2원 상승)
    sell_price = buy_price + 2
    
    print(f"매수 가격: {buy_price}원")
    print(f"목표 판매 가격: {sell_price}원")
    
    # 실제 거래 시작
    print("=== 실제 거래 시작 ===")
    
    # 매수 주문 실행
    buy_amount = 10000  # 1만원 매수
    buy_result = upbit.buy_market_order(coin, buy_amount)
    print(f"매수 주문 결과: {buy_result}")
    
    print(f"매수 완료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 매수 후 잔고 확인
    balance = upbit.get_balance(coin)
    print(f"보유 {coin}: {balance}")
    
    # 목표 가격에 도달할 때까지 모니터링
    while True:
        current_price = pyupbit.get_current_price(coin)
        print(f"현재 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"현재 가격: {current_price}원")
        
        if current_price >= sell_price:
            print(f"목표 가격 도달: {current_price}원")
            
            # 매도 주문
            sell_result = upbit.sell_market_order(coin, balance)
            print(f"매도 주문 결과: {sell_result}")
            
            print(f"매도 완료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=== 거래 종료 ===")
            print(f"매수 가격: {buy_price}원")
            print(f"매도 가격: {current_price}원")
            print(f"예상 수익: {(current_price - buy_price) * balance}원")
            break
        
        # 10초 대기 후 다시 확인
        time.sleep(10)

# 실제 거래 실행
if __name__ == "__main__":
    trading_real()