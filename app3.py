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
        logging.FileHandler("usdc_grid_trade_simulation.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# .env 파일에서 환경변수 로드
load_dotenv()

# 거래 설정
TICKER = "KRW-USDC"  # 유에스디코인 티커
BASE_PRICE = 1400  # 초기 기준 가격 (원)
INIT_AMOUNT = 1000000  # 초기 보유 원화 (100만원)
PRICE_CHANGE = 2  # 가격 변동 기준 (원)
ORDER_AMOUNT = 20000  # 차수별 주문 금액 (원)
MAX_GRID_COUNT = 10  # 최대 분할 매수/매도 차수
SIMULATION_INTERVAL = 10  # 시뮬레이션 간격 (초)

# 전역 변수
current_price = BASE_PRICE  # 현재 가격
wallet = {
    "KRW": INIT_AMOUNT,  # 초기 보유 원화
    "USDC": 0  # 초기 보유 USDC
}
trade_history = []  # 거래 내역 저장 리스트
grid_orders = []  # 그리드 주문 저장 리스트
current_grid_level = 0  # 현재 그리드 레벨


# 그리드 주문 생성
def create_grid_orders():
    """분할 매수/매도 그리드 주문 생성"""
    logger.info("create_grid_orders")
    global grid_orders

    grid_orders = []

    # 기준 가격에서 매수 그리드 생성 (하방)
    for i in range(MAX_GRID_COUNT):
        buy_price = BASE_PRICE - (i * PRICE_CHANGE)
        sell_price = buy_price + PRICE_CHANGE
        volume = ORDER_AMOUNT / buy_price

        grid = {
            'level': i + 1,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'volume': volume,
            'order_amount': ORDER_AMOUNT,
            'status': 'waiting',  # 'waiting', 'bought', 'sold'
            'buy_filled': False,
            'sell_filled': False
        }
        grid_orders.append(grid)

    # 로그 출력
    logger.info(f"총 {len(grid_orders)}개의 그리드 주문 생성됨")
    for grid in grid_orders:
        logger.info(
            f"{grid['level']}차: 매수가 {grid['buy_price']:,.0f}원, 매도가 {grid['sell_price']:,.0f}원, 수량 {grid['volume']:.8f}")

    logger.info("/create_grid_orders")


def show_balance():
    """잔액 정보 출력"""
    logger.info("show_balance")
    logger.info("===== 계정 정보 =====")
    logger.info(f"보유 원화: {wallet['KRW']:,.0f}원")
    logger.info(f"보유 USDC: {wallet['USDC']:.8f} USDC")
    logger.info(f"USDC 현재가: {current_price:,.2f}원")
    if wallet['USDC'] > 0:
        logger.info(f"평가 금액: {wallet['USDC'] * current_price:,.0f}원")

    # 총자산 계산
    total_assets = wallet['KRW'] + (wallet['USDC'] * current_price)
    logger.info(f"총 자산: {total_assets:,.0f}원")
    logger.info("/show_balance")


def buy_usdc(grid_level):
    """지정된 그리드 레벨에서 USDC 매수"""
    logger.info(f"buy_usdc (Level {grid_level})")

    # 해당 그리드 정보 가져오기
    grid = grid_orders[grid_level - 1]

    # 이미 매수한 경우 건너뛰기
    if grid['buy_filled']:
        logger.info(f"레벨 {grid_level}은 이미 매수되었습니다.")
        return False

    # 잔액 확인
    if wallet['KRW'] < grid['order_amount']:
        logger.warning(f"잔액 부족: 매수 불가 (필요: {grid['order_amount']:,}원, 보유: {wallet['KRW']:,}원)")
        return False

    # 매수 수량 계산 (수수료 0.05% 가정)
    volume = (grid['order_amount'] * 0.9995) / grid['buy_price']

    # 거래 실행
    wallet['KRW'] -= grid['order_amount']
    wallet['USDC'] += volume

    # 거래 내역 저장
    trade = {
        'type': 'buy',
        'grid_level': grid_level,
        'price': grid['buy_price'],
        'amount': grid['order_amount'],
        'volume': volume,
        'timestamp': time.time()
    }
    trade_history.append(trade)

    # 그리드 상태 업데이트
    grid['buy_filled'] = True
    grid['status'] = 'bought'

    # 로그 출력
    logger.info(
        f"레벨 {grid_level} USDC 매수 완료: {volume:.8f} USDC (가격: {grid['buy_price']:,.2f}원, 금액: {grid['order_amount']:,}원)")

    show_balance()
    return True


def sell_usdc(grid_level):
    """지정된 그리드 레벨에서 USDC 매도"""
    logger.info(f"sell_usdc (Level {grid_level})")

    # 해당 그리드 정보 가져오기
    grid = grid_orders[grid_level - 1]

    # 매수가 이루어지지 않았거나 이미 매도한 경우 건너뛰기
    if not grid['buy_filled'] or grid['sell_filled']:
        logger.info(f"레벨 {grid_level}은 매도 불가능합니다.")
        return False

    # 매도 금액 계산 (수수료 0.05% 가정)
    volume = grid['volume']
    amount = volume * grid['sell_price'] * 0.9995

    # 보유량 확인 (안전 장치)
    if wallet['USDC'] < volume:
        logger.warning(f"USDC 보유량 부족: 매도 불가 (필요: {volume:.8f}, 보유: {wallet['USDC']:.8f})")
        return False

    # 거래 실행
    wallet['USDC'] -= volume
    wallet['KRW'] += amount

    # 거래 내역 저장
    trade = {
        'type': 'sell',
        'grid_level': grid_level,
        'price': grid['sell_price'],
        'amount': amount,
        'volume': volume,
        'timestamp': time.time()
    }
    trade_history.append(trade)

    # 그리드 상태 업데이트
    grid['sell_filled'] = True
    grid['status'] = 'sold'

    # 수익 계산
    profit = amount - grid['order_amount']
    profit_percentage = (profit / grid['order_amount']) * 100

    # 로그 출력
    logger.info(f"레벨 {grid_level} USDC 매도 완료: {volume:.8f} USDC (가격: {grid['sell_price']:,.2f}원, 금액: {amount:,.0f}원)")
    logger.info(f"레벨 {grid_level} 수익: {profit:+,.0f}원 ({profit_percentage:+.2f}%)")

    # 그리드 초기화 (재사용 가능하도록)
    grid['buy_filled'] = False
    grid['sell_filled'] = False
    grid['status'] = 'waiting'

    show_balance()
    return True


def check_price_and_trade():
    """현재 가격을 확인하고 모든 그리드 주문에 대해 거래 실행"""
    logger.info("check_price_and_trade")
    logger.info(f"현재 USDC 가격: {current_price:,.2f}원")

    # 모든 그리드 주문 확인
    for grid in grid_orders:
        level = grid['level']

        # 매수 조건: 현재 가격이 매수가 이하이고 아직 매수되지 않은 경우
        if current_price <= grid['buy_price'] and not grid['buy_filled']:
            logger.info(f"레벨 {level} 매수 조건 충족: 현재가({current_price:,.2f}원) <= 매수가({grid['buy_price']:,.2f}원)")
            buy_usdc(level)

        # 매도 조건: 현재 가격이 매도가 이상이고 이미 매수되었지만 아직 매도되지 않은 경우
        elif current_price >= grid['sell_price'] and grid['buy_filled'] and not grid['sell_filled']:
            logger.info(f"레벨 {level} 매도 조건 충족: 현재가({current_price:,.2f}원) >= 매도가({grid['sell_price']:,.2f}원)")
            sell_usdc(level)

    logger.info("/check_price_and_trade")


def simulate_price_change(change_amount):
    """
    가격 변동 시뮬레이션

    Args:
        change_amount (float): 변동 금액 (양수: 상승, 음수: 하락)
    """
    logger.info("simulate_price_change")
    global current_price

    previous_price = current_price
    current_price += change_amount

    # 최소 가격 제한
    if current_price < 1:
        current_price = 1

    logger.info(f"[시뮬레이션] USDC 가격 {change_amount:+.2f}원 변동: {previous_price:,.2f}원 -> {current_price:,.2f}원")
    logger.info("/simulate_price_change")


def run_simulation():
    """시뮬레이션 실행"""
    logger.info("run_simulation")
    global current_price

    # 초기 설정
    logger.info("===== USDC 자동 매매 시뮬레이션 시작 =====")
    logger.info(f"초기 기준 가격: {BASE_PRICE:,.2f}원")
    logger.info(f"차수별 주문 금액: {ORDER_AMOUNT:,}원")
    logger.info(f"가격 변동 기준: {PRICE_CHANGE}원")
    logger.info(f"최대 분할 차수: {MAX_GRID_COUNT}")
    logger.info(f"시뮬레이션 간격: {SIMULATION_INTERVAL}초")

    # 그리드 주문 생성
    create_grid_orders()

    # 초기 잔액 출력
    show_balance()

    try:
        # 기본 시나리오: 가격 하락 후 상승
        step_count = 20  # 시뮬레이션 단계 수

        # 초기 확인
        logger.info("\n[시나리오 시작] 현재 가격에서 거래 상태 확인")
        check_price_and_trade()
        time.sleep(SIMULATION_INTERVAL)

        # 가격 하락 시나리오 (단계적으로 하락)
        for i in range(1, 12):
            logger.info(f"\n[시나리오 {i + 1}] 가격 {PRICE_CHANGE}원 하락")
            simulate_price_change(-PRICE_CHANGE)
            check_price_and_trade()
            time.sleep(SIMULATION_INTERVAL)

        # 가격 상승 시나리오 (단계적으로 상승)
        for i in range(12, 24):
            logger.info(f"\n[시나리오 {i + 1}] 가격 {PRICE_CHANGE}원 상승")
            simulate_price_change(PRICE_CHANGE)
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

        # 그리드 별 상태 요약
        logger.info("\n그리드 상태 요약:")
        for grid in grid_orders:
            logger.info(
                f"레벨 {grid['level']} - 상태: {grid['status']}, 매수: {'완료' if grid['buy_filled'] else '대기'}, 매도: {'완료' if grid['sell_filled'] else '대기'}")

        # 거래 내역 요약
        buy_count = sum(1 for trade in trade_history if trade['type'] == 'buy')
        sell_count = sum(1 for trade in trade_history if trade['type'] == 'sell')
        logger.info(f"\n총 거래 횟수: {len(trade_history)}회 (매수: {buy_count}회, 매도: {sell_count}회)")

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