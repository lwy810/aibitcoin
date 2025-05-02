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


def buy_bitcoin_10000krw():
    """
    1만원 어치의 비트코인(BTC)을 시장가로 구매하는 함수

    Returns:
        dict: 주문 결과를 담은 딕셔너리
    """
    try:
        # 비트코인 티커
        ticker = "KRW-BTC"

        # 보유 원화 확인
        krw_balance = upbit.get_balance("KRW")
        print(f"보유 원화: {krw_balance:,.0f}원")

        # 잔액이 1만원 이상인지 확인
        if krw_balance < 10000:
            print("잔액이 1만원 미만입니다. 주문을 진행할 수 없습니다.")
            return None

        # 시장가 매수 주문 (1만원 어치)
        order = upbit.buy_market_order(ticker, 10000)

        print("==== 매수 주문 완료 ====")
        print(f"주문 ID: {order['uuid']}")
        print(f"주문 금액: 10,000원")

        # 주문 상태 확인을 위한 대기
        time.sleep(1)

        # 주문 상태 확인
        order_info = upbit.get_order(order['uuid'])
        print(f"주문 상태: {order_info['state']}")

        # 현재 비트코인 잔액 확인
        btc_balance = upbit.get_balance(ticker)
        print(f"보유 비트코인: {btc_balance:.8f} BTC")

        return order

    except Exception as e:
        print(f"매수 주문 중 오류 발생: {e}")
        return None


def sell_all_bitcoin():
    """
    보유한 모든 비트코인(BTC)을 시장가로 판매하는 함수

    Returns:
        dict: 주문 결과를 담은 딕셔너리
    """
    try:
        # 비트코인 티커
        ticker = "KRW-BTC"

        # 보유 비트코인 확인
        btc_balance = upbit.get_balance(ticker)
        print(f"보유 비트코인: {btc_balance:.8f} BTC")

        # 비트코인이 없으면 종료
        if btc_balance <= 0:
            print("판매할 비트코인이 없습니다.")
            return None

        # 시장가 매도 주문 (전량)
        order = upbit.sell_market_order(ticker, btc_balance)

        print("==== 매도 주문 완료 ====")
        print(f"주문 ID: {order['uuid']}")
        print(f"매도 수량: {btc_balance:.8f} BTC")

        # 주문 상태 확인을 위한 대기
        time.sleep(1)

        # 주문 상태 확인
        order_info = upbit.get_order(order['uuid'])
        print(f"주문 상태: {order_info['state']}")

        # 현재 원화 잔액 확인
        krw_balance = upbit.get_balance("KRW")
        print(f"보유 원화: {krw_balance:,.0f}원")

        return order

    except Exception as e:
        print(f"매도 주문 중 오류 발생: {e}")
        return None


# 함수 사용 예시
if __name__ == "__main__":
    while True:
        # 계정 정보 출력
        print("===== 계정 정보 =====")
        krw_balance = upbit.get_balance("KRW")
        btc_balance = upbit.get_balance("KRW-BTC")
        print(f"보유 원화: {krw_balance:,.0f}원")
        print(f"보유 비트코인: {btc_balance:.8f} BTC")

        # 사용자 입력 받기
        print("\n1. 1만원 어치 비트코인 구매")
        print("2. 보유 비트코인 전량 판매")
        print("3. 종료")
        choice = input("\n원하는 기능을 선택하세요 (1/2/3): ")

        if choice == '1':
            buy_bitcoin_10000krw()
        elif choice == '2':
            sell_all_bitcoin()
        else:
            print("프로그램을 종료합니다.")