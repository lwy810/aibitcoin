import os
import time
import logging
import random
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("usdc_trade_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# .env 파일에서 환경변수 로드
load_dotenv()

# 거래 설정
TICKER = "KRW-USDC"  # 유에스디코인 티커
BUY_AMOUNT = 10000  # 매수 금액 (원)
PRICE_CHANGE = 2  # 가격 변동 기준 (원)
SIMULATION_INTERVAL = 10  # 시뮬레이션 간격 (초)
INIT_AMOUNT = 1000000 # 초기 보유 원화 (예시: 100만원)

# 전역 변수
base_price = 1000  # 초기 기준 가격 (예시: 1000원)
current_price = base_price  # 현재 가격
wallet = {
    "KRW": INIT_AMOUNT,  # 초기 보유 원화 (예시: 100만원)
    "USDC": 0  # 초기 보유 USDC (0개)
}
trade_history = []  # 거래 내역 저장 리스트


def show_balance():
    """잔액 정보 출력"""
    logger.info("show_balance")
    logger.info("===== 계정 정보 =====")
    logger.info(f"보유 원화: {wallet['KRW']:,.0f}원")
    logger.info(f"보유 USDC: {wallet['USDC']:.4f} USDC")
    logger.info(f"USDC 현재가: {current_price:,.2f}원")
    if wallet['USDC'] > 0:
        logger.info(f"평가 금액: {wallet['USDC'] * current_price:,.0f}원")
    logger.info("/show_balance")


def buy_usdc():
    """USDC 매수 시뮬레이션"""
    logger.info("buy_usdc")
    global base_price

    # 잔액 확인
    if wallet['KRW'] < BUY_AMOUNT:
        logger.warning(f"잔액 부족: 매수 불가 (필요: {BUY_AMOUNT:,}원, 보유: {wallet['KRW']:,}원)")
        return False

    # 매수 수량 계산 (수수료 0.05% 가정)
    volume = (BUY_AMOUNT * 0.9995) / current_price

    # 거래 실행
    wallet['KRW'] -= BUY_AMOUNT
    wallet['USDC'] += volume

    # 거래 내역 저장
    trade = {
        'type': 'buy',
        'price': current_price,
        'amount': BUY_AMOUNT,
        'volume': volume,
        'timestamp': time.time()
    }
    trade_history.append(trade)

    # 로그 출력
    logger.info(f"USDC 매수 완료: {volume:.4f} USDC (가격: {current_price:,.2f}원, 금액: {BUY_AMOUNT:,}원)")

    # 기준 가격 업데이트
    base_price = current_price
    logger.info("===== 기준 가격 업데이트 =====")
    logger.info(f"새 기준 가격: {base_price:,.2f}원")
    logger.info(f"매도 트리거 가격: {base_price + PRICE_CHANGE:,.2f}원")
    logger.info(f"매수 트리거 가격: {base_price - PRICE_CHANGE:,.2f}원")
    logger.info("===== 기준 가격 업데이트 =====")

    logger.info("/buy_usdc")

    show_balance()
    return True


def sell_usdc():
    """USDC 매도 시뮬레이션"""
    logger.info("sell_usdc")
    global base_price

    # 보유량 확인
    if wallet['USDC'] <= 0:
        logger.warning("판매할 USDC가 없습니다.")
        return False

    # 매도 금액 계산 (수수료 0.05% 가정)
    volume = wallet['USDC']
    amount = volume * current_price * 0.9995

    # 이전 매수 내역 찾기
    buy_trade = None
    for trade in reversed(trade_history):
        if trade['type'] == 'buy':
            buy_trade = trade
            break

    # 거래 실행
    wallet['USDC'] = 0
    wallet['KRW'] += amount

    # 거래 내역 저장
    trade = {
        'type': 'sell',
        'price': current_price,
        'amount': amount,
        'volume': volume,
        'timestamp': time.time()
    }
    trade_history.append(trade)

    # 로그 출력
    logger.info(f"USDC 매도 완료: {volume:.4f} USDC (가격: {current_price:,.2f}원, 금액: {amount:,.0f}원)")

    # 수익 계산
    if buy_trade:
        buy_price = buy_trade['price']
        profit = amount - buy_trade['amount']
        profit_percentage = (profit / buy_trade['amount']) * 100
        logger.info(f"이전 매수가: {buy_price:,.2f}원, 매도가: {current_price:,.2f}원")
        logger.info(f"수익: {profit:+,.0f}원 ({profit_percentage:+.2f}%)")

    # 기준 가격 업데이트
    base_price = current_price
    logger.info("=================================")
    logger.info(f"새 기준 가격: {base_price:,.2f}원")
    logger.info(f"매도 트리거 가격: {base_price + PRICE_CHANGE:,.2f}원")
    logger.info(f"매수 트리거 가격: {base_price - PRICE_CHANGE:,.2f}원")
    logger.info("=================================")

    logger.info("/sell_usdc")

    show_balance()
    return True


def check_price_and_trade():
    """현재 가격을 확인하고 거래 조건 충족 시 매수/매도"""
    logger.info("check_price_and_trade")
    global base_price

    logger.info(f"현재 USDC 가격: {current_price:,.2f}원 (기준가: {base_price:,.2f}원)")
    logger.info(f"매수 트리거: {base_price - PRICE_CHANGE:,.2f}원, 매도 트리거: {base_price + PRICE_CHANGE:,.2f}원")

    # 매수/매도 조건 확인
    if current_price >= base_price + PRICE_CHANGE:
        # 1원 이상 올랐을 때
        if wallet['USDC'] > 0:
            logger.info(f"매도 조건 충족: 현재가({current_price:,.2f}원) >= 기준가+1원({base_price + PRICE_CHANGE:,.2f}원)")
            sell_usdc()
        else:
            logger.info("매도 조건 충족: 보유 USDC가 없어 매도 주문이 없습니다.")

    elif current_price <= base_price - PRICE_CHANGE:
        # 1원 이상 내렸을 때
        logger.info(f"매수 조건 충족: 현재가({current_price:,.2f}원) <= 기준가-1원({base_price - PRICE_CHANGE:,.2f}원)")
        buy_usdc()

    else:
        # 가격이 기준 범위 내에 있는 경우
        logger.info(f"가격이 기준 범위 내에 있습니다. 매수/매도 대기중...")

    logger.info("/check_price_and_trade")

def simulate_price_change(scenario):
    """
    가격 변동 시나리오 시뮬레이션

    Args:
        scenario (str): 시나리오 유형 ('up1', 'down1', 'down2')
    """
    logger.info("simulate_price_change")
    global current_price

    if scenario == 'up1':
        # 1원 상승 시나리오
        change = PRICE_CHANGE  # 정확히 1원보다 약간 더 올림
        current_price += change
        logger.info(f"[시뮬레이션] USDC 가격 {change:+.2f}원 변동: {current_price:,.2f}원")

    elif scenario == 'down1':
        # 1원 하락 시나리오
        change = -(PRICE_CHANGE )  # 정확히 1원보다 약간 더 내림
        current_price += change
        logger.info(f"[시뮬레이션] USDC 가격 {change:+.2f}원 변동: {current_price:,.2f}원")

    logger.info("/simulate_price_change")


def run_simulation():
    """시뮬레이션 실행"""
    logger.info("run_simulation")
    global base_price, current_price

    # 초기 설정
    logger.info("===== USDC 자동 매매 시뮬레이션 시작 =====")
    logger.info(f"초기 기준 가격: {base_price:,.2f}원")
    logger.info(f"매수 금액: {BUY_AMOUNT:,}원")
    logger.info(f"가격 변동 기준: {PRICE_CHANGE}원")
    logger.info(f"시뮬레이션 간격: {SIMULATION_INTERVAL}초")


    # 현재 가격에서 첫번 매수
    logger.info("\n\n===== 현재 가격에서 첫번 매수 =====")
    buy_usdc()

    # show_balance()

    try:
        # 시나리오 1: 현재 가격에서 시작
        logger.info("\n\n[시나리오 시작] 현재 가격에서 거래 시작")
        check_price_and_trade()
        time.sleep(SIMULATION_INTERVAL)

        # 시나리오 2: 가격 1원 상승
        logger.info("\n[시나리오 2] 가격 1원 상승")
        simulate_price_change('up1')
        check_price_and_trade()
        time.sleep(SIMULATION_INTERVAL)

        # 시나리오 3: 가격 1원 하락 (초기 가격으로 돌아옴)
        logger.info("\n[시나리오 3] 가격 1원 하락")
        simulate_price_change('down1')
        check_price_and_trade()
        time.sleep(SIMULATION_INTERVAL)


        # 최종 결과 출력
        logger.info("\n\n===== 시뮬레이션 완료 =====")
        show_balance()

        # 손익 계산
        initial_balance = INIT_AMOUNT  # 초기 원화 잔액
        final_balance = wallet['KRW'] + (wallet['USDC'] * current_price)
        profit = final_balance - initial_balance
        profit_percentage = (profit / initial_balance) * 100

        logger.info(f"\n\n시뮬레이션 결과:")
        logger.info(f"초기 자산: {initial_balance:,.0f}원")
        logger.info(f"최종 자산: {final_balance:,.0f}원")
        logger.info(f"손익: {profit:+,.0f}원 ({profit_percentage:+.2f}%)")

    except KeyboardInterrupt:
        logger.info("\n사용자에 의해 시뮬레이션이 중단되었습니다.")
    except Exception as e:
        logger.error(f"\n시뮬레이션 중 오류 발생: {str(e)}")
    finally:
        logger.info("시뮬레이션이 종료되었습니다.")

    logger.info("/run_simulation")

# 시뮬레이션 실행
if __name__ == "__main__":
    logger.info("__main__")
    run_simulation()