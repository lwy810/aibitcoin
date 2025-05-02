import os
import time
import pyupbit
from dotenv import load_dotenv
import logging
import schedule

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("usdc_trade.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# .env 파일에서 환경변수 로드
load_dotenv()

# 업비트 API 키 설정
access_key = os.environ.get("UPBIT_ACCESS_KEY")
secret_key = os.environ.get("UPBIT_SECRET_KEY")

# 업비트 객체 생성
upbit = pyupbit.Upbit(access_key, secret_key)

# 거래 설정
TICKER = "KRW-USDC"  # 유에스디코인 티커
BUY_AMOUNT = 20000  # 매수 금액 (원) - 최소 주문 금액 증가
PRICE_RISE = 1  # 매도 트리거 가격 차이 (원)
USE_LIMIT_ORDER = True  # 지정가 주문 사용 여부

# 전역 변수
bought = False  # 매수 여부
buy_price = 0  # 매수 가격
buy_volume = 0  # 매수 수량


def show_balance():
    """
    계정 잔액 정보를 출력
    """
    krw_balance = upbit.get_balance("KRW")
    usdc_balance = upbit.get_balance(TICKER)
    current_price = pyupbit.get_current_price(TICKER)

    logger.info("===== 계정 정보 =====")
    logger.info(f"보유 원화: {krw_balance:,.0f}원")
    logger.info(f"보유 USDC: {usdc_balance:.4f} USDC")
    if current_price:
        logger.info(f"USDC 현재가: {current_price:,.2f}원")
        if usdc_balance > 0:
            logger.info(f"평가 금액: {usdc_balance * current_price:,.0f}원")


def buy_usdc():
    """
    USDC 매수 함수 - 지정가 주문 또는 시장가 주문 사용
    """
    global bought, buy_price, buy_volume

    try:
        # 이미 매수했는지 확인
        if bought:
            logger.info("이미 USDC를 매수한 상태입니다.")
            return

        # 잔액 확인
        krw_balance = upbit.get_balance("KRW")
        if krw_balance < BUY_AMOUNT:
            logger.warning(f"잔액 부족: 매수 불가 (필요: {BUY_AMOUNT:,}원, 보유: {krw_balance:,}원)")
            return

        # 현재가 확인
        current_price = pyupbit.get_current_price(TICKER)
        if not current_price:
            logger.warning("현재 가격 조회 실패")
            return

        # 호가 정보 확인 (지정가 주문용)
        orderbook = pyupbit.get_orderbook(TICKER)
        if USE_LIMIT_ORDER and orderbook:
            # 최우선 매도호가로 지정가 주문 설정
            ask_price = orderbook['orderbook_units'][0]['ask_price']

            # 수량 계산 (수수료 고려하여 약간 적게)
            volume = (BUY_AMOUNT * 0.9995) / ask_price

            logger.info(f"USDC 지정가 매수 시도: 가격 {ask_price:,.2f}원, 수량 {volume:.4f}, 금액 {BUY_AMOUNT:,}원")

            # 지정가 매수 주문
            order = upbit.buy_limit_order(TICKER, ask_price, volume)
        else:
            # 시장가 매수 주문
            logger.info(f"USDC 시장가 매수 시도: 현재가 {current_price:,.2f}원, 매수 금액 {BUY_AMOUNT:,}원")
            order = upbit.buy_market_order(TICKER, BUY_AMOUNT)

        if not order or 'uuid' not in order:
            logger.error("매수 주문 실패")
            return

        logger.info(f"매수 주문 완료: 주문번호 {order['uuid']}")

        # 주문 체결 대기
        for _ in range(15):  # 15번 시도 (30초로 증가)
            time.sleep(2)  # 2초 대기
            order_info = upbit.get_order(order['uuid'])

            if not order_info:
                logger.warning("주문 정보 조회 실패")
                continue

            if order_info['state'] == 'done':
                # 매수 정보 저장
                trades = upbit.get_order(order['uuid'])
                total_volume = 0
                total_price = 0

                if 'trades' in trades and trades['trades']:
                    for trade in trades['trades']:
                        volume = float(trade['volume'])
                        price = float(trade['price'])
                        total_volume += volume
                        total_price += volume * price

                average_price = total_price / total_volume if total_volume > 0 else 0

                buy_volume = total_volume
                buy_price = average_price
                bought = True

                logger.info(f"USDC 매수 완료: {buy_volume:.4f} USDC (평균가: {buy_price:,.2f}원)")
                logger.info(f"매도 목표가: {buy_price + PRICE_RISE:,.2f}원")
                show_balance()
                return
            elif order_info['state'] == 'wait':
                logger.info(f"주문 체결 대기 중... (상태: wait)")
            else:
                logger.info(f"주문 체결 대기 중... (상태: {order_info['state']})")

        # 30초 후에도 체결되지 않으면 취소
        logger.warning("주문 체결 시간 초과, 주문을 취소합니다.")
        upbit.cancel_order(order['uuid'])

    except Exception as e:
        logger.error(f"매수 중 오류 발생: {str(e)}")


def sell_usdc():
    """
    보유한 USDC 전량 매도 - 지정가 또는 시장가 주문 사용
    """
    global bought, buy_price, buy_volume

    try:
        # 매수 여부 확인
        if not bought:
            logger.info("매수한 USDC가 없습니다.")
            return

        # USDC 잔액 확인
        usdc_balance = upbit.get_balance(TICKER)
        if usdc_balance <= 0:
            logger.warning("판매할 USDC가 없습니다.")
            bought = False  # 상태 초기화
            return

        # 현재가 확인
        current_price = pyupbit.get_current_price(TICKER)
        if not current_price:
            logger.warning("현재 가격 조회 실패")
            return

        # 호가 정보 확인 (지정가 주문용)
        orderbook = pyupbit.get_orderbook(TICKER)
        if USE_LIMIT_ORDER and orderbook:
            # 최우선 매수호가로 지정가 주문 설정
            bid_price = orderbook['orderbook_units'][0]['bid_price']

            logger.info(f"USDC 지정가 매도 시도: 가격 {bid_price:,.2f}원, 수량 {usdc_balance:.4f} USDC")

            # 지정가 매도 주문
            order = upbit.sell_limit_order(TICKER, bid_price, usdc_balance)
        else:
            # 시장가 매도 주문
            logger.info(f"USDC 시장가 매도 시도: 현재가 {current_price:,.2f}원, 매도량 {usdc_balance:.4f} USDC")
            order = upbit.sell_market_order(TICKER, usdc_balance)

        if not order or 'uuid' not in order:
            logger.error("매도 주문 실패")
            return

        logger.info(f"매도 주문 완료: 주문번호 {order['uuid']}")

        # 주문 체결 대기
        for _ in range(15):  # 15번 시도 (30초로 증가)
            time.sleep(2)  # 2초 대기
            order_info = upbit.get_order(order['uuid'])

            if not order_info:
                logger.warning("주문 정보 조회 실패")
                continue

            if order_info['state'] == 'done':
                # 매도 정보 계산
                trades = upbit.get_order(order['uuid'])
                total_volume = 0
                total_price = 0

                if 'trades' in trades and trades['trades']:
                    for trade in trades['trades']:
                        volume = float(trade['volume'])
                        price = float(trade['price'])
                        total_volume += volume
                        total_price += volume * price

                sell_price = total_price / total_volume if total_volume > 0 else 0
                profit = total_price - (buy_price * total_volume)
                profit_percentage = (profit / (buy_price * total_volume)) * 100 if buy_price > 0 else 0

                logger.info(f"USDC 매도 완료: {total_volume:.4f} USDC (매도가: {sell_price:,.2f}원)")
                logger.info(f"매수가: {buy_price:,.2f}원, 매도가: {sell_price:,.2f}원")
                logger.info(f"수익: {profit:,.0f}원 ({profit_percentage:.2f}%)")

                # 상태 초기화
                bought = False
                buy_price = 0
                buy_volume = 0

                show_balance()
                return
            elif order_info['state'] == 'wait':
                logger.info(f"주문 체결 대기 중... (상태: wait)")
            else:
                logger.info(f"주문 체결 대기 중... (상태: {order_info['state']})")

        # 30초 후에도 체결되지 않으면 취소
        logger.warning("주문 체결 시간 초과, 주문을 취소합니다.")
        upbit.cancel_order(order['uuid'])

    except Exception as e:
        logger.error(f"매도 중 오류 발생: {str(e)}")


def check_prices():
    """
    현재 가격을 확인하고 매수/매도 조건을 확인
    """
    try:
        current_price = pyupbit.get_current_price(TICKER)
        if not current_price:
            logger.warning("현재 가격 조회 실패")
            return

        logger.info(f"현재 USDC 가격: {current_price:,.2f}원")

        # 매수 상태 확인
        if not bought:
            logger.info("USDC를 매수합니다.")
            buy_usdc()
        # 매도 조건 확인
        elif current_price >= buy_price + PRICE_RISE:
            logger.info(f"매도 조건 충족: 현재가({current_price:,.2f}원) >= 목표가({buy_price + PRICE_RISE:,.2f}원)")
            sell_usdc()
        else:
            logger.info(f"매도 대기 중: 현재가({current_price:,.2f}원) < 목표가({buy_price + PRICE_RISE:,.2f}원)")

    except Exception as e:
        logger.error(f"가격 확인 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    try:
        logger.info("USDC 자동 매매 프로그램을 시작합니다.")
        show_balance()

        # 첫 번째 가격 확인 및 매수/매도
        check_prices()

        # 10초마다 가격 확인하는 스케줄러 설정
        schedule.every(10).seconds.do(check_prices)

        # 스케줄러 실행
        logger.info("10초마다 가격을 확인합니다. 종료하려면 Ctrl+C를 누르세요.")
        while True:
            schedule.run_pending()
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("프로그램이 종료되었습니다.")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")