import os
import time
import logging
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import pyupbit

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("btc_grid_trade.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# .env 파일에서 환경변수 로드
load_dotenv()

# API 키 설정
ACCESS_KEY = os.environ.get("UPBIT_ACCESS_KEY")
SECRET_KEY = os.environ.get("UPBIT_SECRET_KEY")

# 거래 설정
TICKER = "KRW-USDT"  # 비트코인 티커
BASE_PRICE = None  # 초기 기준 가격 (시장 가격으로 설정됨)
PRICE_CHANGE = 1  # 가격 변동 기준 (원)
ORDER_AMOUNT = 6000  # 차수별 주문 금액 (원)
MAX_GRID_COUNT = 10  # 최대 분할 매수/매도 차수
CHECK_INTERVAL = 10  # 가격 확인 간격 (초)
CANCEL_TIMEOUT = 3600  # 미체결 주문 취소 시간 (초, 1시간)
TEST_MODE = True  # 테스트 모드 활성화 여부
FEE_RATE = 0.0005  # 거래 수수료 (0.05%)

# 전역 변수
current_price = 0  # 현재 가격
grid_orders = []  # 그리드 주문 저장 리스트
trade_history = []  # 거래 내역 저장 리스트
active_orders = {}  # 활성 주문 관리: {uuid: {'grid_level': n, 'type': 'buy'/'sell', 'timestamp': datetime}}

# 가상 잔고 정보를 저장할 전역 변수
virtual_balance = {
    "KRW": 1000000,  # 초기 100만원
    "BTC": 0,        # 초기 0 BTC
    "BTC_AVG_PRICE": 0  # 초기 평균 매수가 0원
}

# Upbit 객체 생성
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)


def get_current_price():
    """현재 가격 조회"""
    try:
        ticker_price = pyupbit.get_current_price(TICKER)
        logger.info(f"현재 {TICKER} 가격: {ticker_price:,.2f}원")
        return ticker_price
    except Exception as e:
        logger.error(f"가격 조회 중 오류 발생: {str(e)}")
        return None


def get_balance():
    """계좌 잔고 조회"""
    logger.info("get_balance")
    global virtual_balance

    try:
        if TEST_MODE:
            # 테스트 모드일 경우 가상 잔고 사용
            krw_balance = virtual_balance["KRW"]
            btc_balance = virtual_balance["BTC"]
            btc_avg_price = virtual_balance["BTC_AVG_PRICE"] if btc_balance > 0 else 0

            logger.info(f"[테스트 모드] 보유 원화: {krw_balance:,.0f}원")
            logger.info(f"[테스트 모드] 보유 BTC: {btc_balance:.8f} BTC")

            if btc_balance > 0:
                logger.info(f"[테스트 모드] BTC 평균 매수가: {btc_avg_price:,.2f}원")
                logger.info(f"[테스트 모드] 평가 금액: {btc_balance * current_price:,.0f}원")

            # 총자산 계산
            total_assets = krw_balance + (btc_balance * current_price)
            logger.info(f"[테스트 모드] 총 자산: {total_assets:,.0f}원")

            return {
                "KRW": krw_balance,
                "BTC": btc_balance,
                "BTC_AVG_PRICE": btc_avg_price,
                "TOTAL_ASSETS": total_assets
            }
        else:
            # 실제 API 사용 (기존 코드)
            # 원화 잔고 조회
            krw_balance = upbit.get_balance("KRW")
            logger.info(f"보유 원화: {krw_balance:,.0f}원")

            # BTC 잔고 조회
            btc_balance = upbit.get_balance(TICKER)
            btc_avg_price = upbit.get_avg_buy_price(TICKER)
            logger.info(f"보유 BTC: {btc_balance:.8f} {TICKER}")

            if btc_balance > 0:
                logger.info(f"{TICKER} 평균 매수가: {btc_avg_price:,.2f}원")
                logger.info(f"평가 금액: {btc_balance * current_price:,.0f}원")

            # 총자산 계산
            total_assets = krw_balance + (btc_balance * current_price)
            logger.info(f"총 자산: {total_assets:,.0f}원")

            return {
                "KRW": krw_balance,
                "BTC": btc_balance,
                "BTC_AVG_PRICE": btc_avg_price,
                "TOTAL_ASSETS": total_assets
            }
    except Exception as e:
        logger.error(f"잔고 조회 중 오류 발생: {str(e)}")
        return None

def create_grid_orders(input_base_price=None):
    """분할 매수/매도 그리드 주문 생성"""
    logger.info("create_grid_orders")
    global grid_orders, BASE_PRICE

    # 입력받은 기준 가격이 있으면 사용
    if input_base_price is not None:
        BASE_PRICE = input_base_price
        logger.info(f"사용자 지정 기준 가격: {BASE_PRICE:,.2f}원")
    # 기준 가격이 없으면 현재 가격으로 설정
    elif BASE_PRICE is None:
        BASE_PRICE = get_current_price()
        if BASE_PRICE is None:
            logger.error("기준 가격을 설정할 수 없습니다. 프로그램을 종료합니다.")
            return False
        logger.info(f"현재 시장 가격으로 기준 가격 설정: {BASE_PRICE:,.2f}원")

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
            'sell_filled': False,
            'buy_order_id': None,
            'sell_order_id': None,
            'buy_order_time': None,
            'sell_order_time': None
        }
        grid_orders.append(grid)

    # 로그 출력
    logger.info(f"총 {len(grid_orders)}개의 그리드 주문 생성됨")
    for grid in grid_orders:
        logger.info(
            f"{grid['level']}차: 매수가 {grid['buy_price']:,.2f}원, 매도가 {grid['sell_price']:,.2f}원, 수량 {grid['volume']:.8f}")

    return True


def buy_btc(grid_level):
    """지정된 그리드 레벨에서 BTC 매수"""
    logger.info(f"buy_btc (Level {grid_level})")
    global virtual_balance

    # 해당 그리드 정보 가져오기
    grid = grid_orders[grid_level - 1]

    # 이미 매수한 경우 건너뛰기
    if grid['buy_filled']:
        logger.info(f"레벨 {grid_level}은 이미 매수되었습니다.")
        return False

    # 이미 매수 주문이 있는 경우
    if grid['buy_order_id']:
        logger.info(f"레벨 {grid_level}은 이미 매수 주문이 진행 중입니다: {grid['buy_order_id']}")
        return False

    try:
        # 잔액 확인
        if TEST_MODE:
            krw_balance = virtual_balance["KRW"]
        else:
            krw_balance = upbit.get_balance("KRW")

        if krw_balance < grid['order_amount']:
            logger.warning(f"잔액 부족: 매수 불가 (필요: {grid['order_amount']:,}원, 보유: {krw_balance:,}원)")
            return False

        # 매수 수량 계산 (수수료 고려)
        price = grid['buy_price']
        order_amount = grid['order_amount']
        volume_without_fee = order_amount / price
        fee_amount = order_amount * FEE_RATE
        actual_volume = volume_without_fee * (1 - FEE_RATE)  # 수수료를 고려한 실제 수량

        if not TEST_MODE:
            # 실제 주문 실행
            order = upbit.buy_limit_order(TICKER, price, volume_without_fee)
            if order and 'uuid' in order:
                uuid = order['uuid']
                grid['buy_order_id'] = uuid
                grid['buy_order_time'] = datetime.now()

                # 활성 주문 목록에 추가
                active_orders[uuid] = {
                    'grid_level': grid_level,
                    'type': 'buy',
                    'price': price,
                    'volume': volume_without_fee,
                    'timestamp': datetime.now()
                }

                logger.info(f"매수 주문 완료: 주문번호 {uuid}")
                return True
            else:
                logger.error(f"매수 주문 실패: {order}")
                return False
        else:
            # 테스트용: 주문 생성 시뮬레이션
            uuid = f"buy-{grid_level}-{int(time.time())}"
            grid['buy_order_id'] = uuid
            grid['buy_order_time'] = datetime.now()

            # 활성 주문 목록에 추가
            active_orders[uuid] = {
                'grid_level': grid_level,
                'type': 'buy',
                'price': price,
                'volume': volume_without_fee,
                'timestamp': datetime.now()
            }

            logger.info(f"매수 주문 완료: 주문번호 {uuid}")

            # 테스트용: 즉시 체결된 것으로 처리
            grid['buy_filled'] = True
            grid['status'] = 'bought'

            # 거래 내역 저장
            trade = {
                'type': 'buy',
                'grid_level': grid_level,
                'price': price,
                'amount': order_amount,
                'volume': actual_volume,  # 수수료 적용된 수량
                'fee': fee_amount,  # 수수료 금액
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'order_id': uuid
            }
            trade_history.append(trade)

            # 활성 주문 목록에서 제거 (테스트용 - 즉시 체결 시)
            if uuid in active_orders:
                del active_orders[uuid]

            # 테스트 모드에서 가상 잔고 업데이트
            virtual_balance["KRW"] -= order_amount
            new_btc = virtual_balance["BTC"] + actual_volume
            new_avg_price = ((virtual_balance["BTC"] * virtual_balance["BTC_AVG_PRICE"]) + (
                        actual_volume * price)) / new_btc if new_btc > 0 else 0
            virtual_balance["BTC"] = new_btc
            virtual_balance["BTC_AVG_PRICE"] = new_avg_price

            # 로그 출력
            logger.info(
                f"레벨 {grid_level} {TICKER} 매수 완료: {actual_volume:.8f} {TICKER} (가격: {price:,.2f}원, 금액: {order_amount:,}원, 수수료: {fee_amount:,.2f}원)")

            # 잔고 출력
            get_balance()
            return True

    except Exception as e:
        logger.error(f"매수 중 오류 발생: {str(e)}")
        return False


def sell_btc(grid_level):
    """지정된 그리드 레벨에서 BTC 매도"""
    logger.info(f"sell_btc (Level {grid_level})")
    global virtual_balance

    # 해당 그리드 정보 가져오기
    grid = grid_orders[grid_level - 1]

    # 매수가 이루어지지 않았거나 이미 매도한 경우 건너뛰기
    if not grid['buy_filled'] or grid['sell_filled']:
        logger.info(f"레벨 {grid_level}은 매도 불가능합니다.")
        return False

    # 이미 매도 주문이 있는 경우
    if grid['sell_order_id']:
        logger.info(f"레벨 {grid_level}은 이미 매도 주문이 진행 중입니다: {grid['sell_order_id']}")
        return False

    try:
        # 보유량 확인
        if TEST_MODE:
            btc_balance = virtual_balance["BTC"]
        else:
            btc_balance = upbit.get_balance(TICKER)

        volume = grid['volume'] * (1 - FEE_RATE)  # 수수료 적용된 수량
        if btc_balance < volume:
            logger.warning(f"{TICKER} 보유량 부족: 매도 불가 (필요: {volume:.8f}, 보유: {btc_balance:.8f})")
            return False

        # 매도 가격
        price = grid['sell_price']

        if not TEST_MODE:
            # 실제 주문 실행
            order = upbit.sell_limit_order(TICKER, price, volume)
            if order and 'uuid' in order:
                uuid = order['uuid']
                grid['sell_order_id'] = uuid
                grid['sell_order_time'] = datetime.now()

                # 활성 주문 목록에 추가
                active_orders[uuid] = {
                    'grid_level': grid_level,
                    'type': 'sell',
                    'price': price,
                    'volume': volume,
                    'timestamp': datetime.now()
                }

                logger.info(f"매도 주문 완료: 주문번호 {uuid}")
                return True
            else:
                logger.error(f"매도 주문 실패: {order}")
                return False
        else:
            # 테스트용: 주문 생성 시뮬레이션
            uuid = f"sell-{grid_level}-{int(time.time())}"
            grid['sell_order_id'] = uuid
            grid['sell_order_time'] = datetime.now()

            # 활성 주문 목록에 추가
            active_orders[uuid] = {
                'grid_level': grid_level,
                'type': 'sell',
                'price': price,
                'volume': volume,
                'timestamp': datetime.now()
            }

            logger.info(f"매도 주문 완료: 주문번호 {uuid}")

            # 테스트용: 즉시 체결된 것으로 처리
            grid['sell_filled'] = True
            grid['status'] = 'sold'

            # 수수료 계산
            sell_amount = volume * price
            fee_amount = sell_amount * FEE_RATE
            amount = sell_amount - fee_amount  # 수수료를 제외한 실제 매도 금액

            # 테스트 모드에서 가상 잔고 업데이트
            virtual_balance["KRW"] += amount
            virtual_balance["BTC"] -= volume

            # BTC가 0이 되면 평균 매수가도 0으로 초기화
            if virtual_balance["BTC"] <= 0:
                virtual_balance["BTC"] = 0
                virtual_balance["BTC_AVG_PRICE"] = 0

            # 거래 내역 저장
            trade = {
                'type': 'sell',
                'grid_level': grid_level,
                'price': price,
                'amount': amount,
                'volume': volume,
                'fee': fee_amount,  # 수수료 금액
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'order_id': uuid
            }
            trade_history.append(trade)

            # 활성 주문 목록에서 제거 (테스트용 - 즉시 체결 시)
            if uuid in active_orders:
                del active_orders[uuid]

            # 수익 계산
            profit = amount - grid['order_amount']
            profit_percentage = (profit / grid['order_amount']) * 100

            # 로그 출력
            logger.info(
                f"레벨 {grid_level} {TICKER} 매도 완료: {volume:.8f} {TICKER} (가격: {price:,.2f}원, 금액: {amount:,.0f}원, 수수료: {fee_amount:,.2f}원)")
            logger.info(f"레벨 {grid_level} 수익: {profit:+,.0f}원 ({profit_percentage:+.2f}%)")

            # 그리드 초기화 (재사용 가능하도록)
            grid['buy_filled'] = False
            grid['sell_filled'] = False
            grid['status'] = 'waiting'
            grid['buy_order_id'] = None
            grid['sell_order_id'] = None
            grid['buy_order_time'] = None
            grid['sell_order_time'] = None

            # 잔고 출력
            get_balance()
            return True

    except Exception as e:
        logger.error(f"매도 중 오류 발생: {str(e)}")
        return False

def cancel_order(uuid):
    """주문 취소"""
    logger.info(f"cancel_order: {uuid}")
    global virtual_balance

    try:
        if not TEST_MODE:
            result = upbit.cancel_order(uuid)
            if result and 'uuid' in result:
                logger.info(f"주문 취소 성공: {uuid}")
                return True
            else:
                logger.error(f"주문 취소 실패: {result}")
                return False
        else:
            # 테스트용: 주문 취소 시뮬레이션
            logger.info(f"주문 취소 성공: {uuid}")

            # 활성 주문 목록에서 정보 가져온 후 제거
            if uuid in active_orders:
                order_info = active_orders[uuid]
                grid_level = order_info['grid_level']
                order_type = order_info['type']
                price = order_info['price']
                volume = order_info['volume']

                # 해당 그리드 정보 가져오기
                grid = grid_orders[grid_level - 1]

                # 그리드 상태 업데이트
                if order_type == 'buy':
                    grid['buy_order_id'] = None
                    grid['buy_order_time'] = None
                else:  # sell
                    grid['sell_order_id'] = None
                    grid['sell_order_time'] = None

                # 활성 주문 목록에서 제거
                del active_orders[uuid]

                # 거래 내역에 취소 기록 추가
                trade = {
                    'type': f"{order_type}_cancel",
                    'grid_level': grid_level,
                    'price': order_info['price'],
                    'volume': order_info['volume'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'order_id': uuid
                }
                trade_history.append(trade)

                logger.info(f"레벨 {grid_level}의 {order_type} 주문이 취소되었습니다.")

            return True

    except Exception as e:
        logger.error(f"주문 취소 중 오류 발생: {str(e)}")
        return False

def check_orders():
    """주문 상태 확인 및 업데이트"""
    logger.info("check_orders")

    # 현재 시간
    now = datetime.now()

    # 활성 주문 목록 복사 (for문에서 삭제 시 에러 방지)
    active_order_uuids = list(active_orders.keys())

    for uuid in active_order_uuids:
        order_info = active_orders[uuid]
        grid_level = order_info['grid_level']
        order_type = order_info['type']
        order_time = order_info['timestamp']

        # 해당 그리드 정보 가져오기
        grid = grid_orders[grid_level - 1]

        # 1. 주문 상태 확인
        try:
            if not TEST_MODE:
                order = upbit.get_order(uuid)

                # 체결된 경우
                if order['state'] == 'done':
                    if order_type == 'buy':
                        grid['buy_filled'] = True
                        grid['status'] = 'bought'
                    else:  # sell
                        grid['sell_filled'] = True
                        grid['status'] = 'sold'

                        # 그리드 초기화 (매도 완료 후)
                        if order_type == 'sell':
                            grid['buy_filled'] = False
                            grid['sell_filled'] = False
                            grid['status'] = 'waiting'
                            grid['buy_order_id'] = None
                            grid['sell_order_id'] = None
                            grid['buy_order_time'] = None
                            grid['sell_order_time'] = None

                    # 활성 주문 목록에서 제거
                    del active_orders[uuid]
                    logger.info(f"레벨 {grid_level}의 {order_type} 주문이 체결되었습니다: {uuid}")

                    # 잔고 출력
                    get_balance()
                else:
                    # 테스트용 - 실제 API 호출이 없으므로 이 부분 생략
                    pass

        except Exception as e:
            logger.error(f"주문 상태 확인 중 오류: {str(e)}")

        # 2. 시간 초과 주문 취소 (CANCEL_TIMEOUT 초 이상 체결되지 않은 주문)
        time_diff = (now - order_time).total_seconds()
        if time_diff > CANCEL_TIMEOUT:
            logger.info(f"시간 초과 주문 발견: {uuid}, 경과 시간: {time_diff:.0f}초")
            cancel_order(uuid)

def check_price_and_trade():
    """현재 가격을 확인하고 모든 그리드 주문에 대해 거래 실행"""
    logger.info("check_price_and_trade")
    global current_price

    # 현재 가격 업데이트
    current_price = get_current_price()
    if current_price is None:
        logger.error("가격 조회 실패")
        return

    # 모든 그리드 주문 확인
    for grid in grid_orders:
        level = grid['level']

        # 매수 조건: 현재 가격이 매수가 이하이고 아직 매수되지 않은 경우 (주문 없는 경우)
        if current_price <= grid['buy_price'] and not grid['buy_filled'] and not grid['buy_order_id']:
            logger.info(f"레벨 {level} 매수 조건 충족: 현재가({current_price:,.2f}원) <= 매수가({grid['buy_price']:,.2f}원)")
            buy_btc(level)

        # 매도 조건: 현재 가격이 매도가 이상이고 이미 매수되었지만 아직 매도되지 않은 경우 (주문 없는 경우)
        elif current_price >= grid['sell_price'] and grid['buy_filled'] and not grid['sell_filled'] and not grid[
            'sell_order_id']:
            logger.info(f"레벨 {level} 매도 조건 충족: 현재가({current_price:,.2f}원) >= 매도가({grid['sell_price']:,.2f}원)")
            sell_btc(level)

def save_trading_results():
    """거래 결과 저장"""
    try:
        # 거래 내역 저장
        with open("btc_trade_history.csv", "w", encoding="utf-8") as f:
            f.write("type,grid_level,price,amount,volume,fee,timestamp,order_id\n")
            for trade in trade_history:
                amount = trade.get('amount', 0)
                fee = trade.get('fee', 0)
                f.write(
                    f"{trade['type']},{trade['grid_level']},{trade['price']},{amount},{trade['volume']},{fee},{trade['timestamp']},{trade.get('order_id', '')}\n")

        # 활성 주문 저장
        with open("btc_active_orders.csv", "w", encoding="utf-8") as f:
            f.write("order_id,type,grid_level,price,volume,timestamp\n")
            for uuid, order in active_orders.items():
                f.write(
                    f"{uuid},{order['type']},{order['grid_level']},{order['price']},{order['volume']},{order['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 그리드 상태 저장
        with open("btc_grid_status.csv", "w", encoding="utf-8") as f:
            f.write("level,buy_price,sell_price,volume,status,buy_filled,sell_filled,buy_order_id,sell_order_id\n")
            for grid in grid_orders:
                f.write(
                    f"{grid['level']},{grid['buy_price']},{grid['sell_price']},{grid['volume']},{grid['status']},{grid['buy_filled']},{grid['sell_filled']},{grid['buy_order_id'] or ''},{grid['sell_order_id'] or ''}\n")

        # 가상 잔고 저장 (테스트 모드인 경우)
        if TEST_MODE:
            with open("btc_virtual_balance.csv", "w", encoding="utf-8") as f:
                f.write("currency,balance,avg_price,timestamp\n")
                f.write(f"KRW,{virtual_balance['KRW']},0,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"BTC,{virtual_balance['BTC']},{virtual_balance['BTC_AVG_PRICE']},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        logger.info(f"거래 데이터가 CSV 파일에 저장되었습니다.")
    except Exception as e:
        logger.error(f"거래 결과 저장 중 오류 발생: {str(e)}")

def cancel_all_orders():
    """모든 주문 취소"""
    logger.info("cancel_all_orders")

    # 활성 주문 목록 복사 (for문에서 삭제 시 에러 방지)
    active_order_uuids = list(active_orders.keys())

    canceled_count = 0
    for uuid in active_order_uuids:
        if cancel_order(uuid):
            canceled_count += 1

    logger.info(f"총 {canceled_count}개의 주문이 취소되었습니다.")
    return canceled_count


def load_trading_state():
    """이전 거래 상태 로드"""
    logger.info("load_trading_state")
    global trade_history, active_orders, grid_orders, BASE_PRICE, virtual_balance

    try:
        # 1. 그리드 상태 로드
        if os.path.exists("btc_grid_status.csv"):
            grid_orders = []
            with open("btc_grid_status.csv", "r", encoding="utf-8") as f:
                lines = f.readlines()[1:]  # 헤더 제외
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 9:
                        level = int(parts[0])
                        buy_price = float(parts[1])
                        sell_price = float(parts[2])
                        volume = float(parts[3])
                        status = parts[4]
                        buy_filled = parts[5].lower() == 'true'
                        sell_filled = parts[6].lower() == 'true'
                        buy_order_id = parts[7] if parts[7] else None
                        sell_order_id = parts[8] if parts[8] else None

                        # 기준 가격 추정 (첫 번째 그리드에서)
                        if level == 1:
                            BASE_PRICE = buy_price

                        grid = {
                            'level': level,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'volume': volume,
                            'order_amount': buy_price * volume,
                            'status': status,
                            'buy_filled': buy_filled,
                            'sell_filled': sell_filled,
                            'buy_order_id': buy_order_id,
                            'sell_order_id': sell_order_id,
                            'buy_order_time': None,
                            'sell_order_time': None
                        }
                        grid_orders.append(grid)

            logger.info(f"그리드 상태 로드 완료: {len(grid_orders)}개의 그리드")

        # 2. 활성 주문 로드
        if os.path.exists("btc_active_orders.csv"):
            active_orders = {}
            with open("btc_active_orders.csv", "r", encoding="utf-8") as f:
                lines = f.readlines()[1:]  # 헤더 제외
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        uuid = parts[0]
                        order_type = parts[1]
                        grid_level = int(parts[2])
                        price = float(parts[3])
                        volume = float(parts[4])
                        timestamp = datetime.strptime(parts[5], '%Y-%m-%d %H:%M:%S')

                        active_orders[uuid] = {
                            'grid_level': grid_level,
                            'type': order_type,
                            'price': price,
                            'volume': volume,
                            'timestamp': timestamp
                        }

                        # 그리드 주문 시간 업데이트
                        if 0 <= grid_level - 1 < len(grid_orders):
                            grid = grid_orders[grid_level - 1]
                            if order_type == 'buy':
                                grid['buy_order_time'] = timestamp
                            else:  # sell
                                grid['sell_order_time'] = timestamp

            logger.info(f"활성 주문 로드 완료: {len(active_orders)}개의 주문")

        # 3. 거래 내역 로드
        if os.path.exists("btc_trade_history.csv"):
            trade_history = []
            with open("btc_trade_history.csv", "r", encoding="utf-8") as f:
                lines = f.readlines()[1:]  # 헤더 제외
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 8:  # fee 필드 추가로 총 8개 필드
                        trade_type = parts[0]
                        grid_level = int(parts[1])
                        price = float(parts[2])
                        amount = float(parts[3]) if parts[3] else 0
                        volume = float(parts[4])
                        fee = float(parts[5]) if parts[5] else 0
                        timestamp = parts[6]
                        order_id = parts[7] if parts[7] else None

                        trade = {
                            'type': trade_type,
                            'grid_level': grid_level,
                            'price': price,
                            'amount': amount,
                            'volume': volume,
                            'fee': fee,
                            'timestamp': timestamp,
                            'order_id': order_id
                        }
                        trade_history.append(trade)

            logger.info(f"거래 내역 로드 완료: {len(trade_history)}개의 거래")

        # 4. 가상 잔고 로드 (테스트 모드인 경우)
        if TEST_MODE and os.path.exists("btc_virtual_balance.csv"):
            with open("btc_virtual_balance.csv", "r", encoding="utf-8") as f:
                lines = f.readlines()[1:]  # 헤더 제외
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        currency = parts[0]
                        balance = float(parts[1])
                        avg_price = float(parts[2])

                        if currency == "KRW":
                            virtual_balance["KRW"] = balance
                        elif currency == "BTC":
                            virtual_balance["BTC"] = balance
                            virtual_balance["BTC_AVG_PRICE"] = avg_price

            logger.info(f"가상 잔고 로드 완료: KRW {virtual_balance['KRW']:,.0f}원, {TICKER} {virtual_balance['BTC']:.8f}")

        return len(grid_orders) > 0

    except Exception as e:
        logger.error(f"거래 상태 로드 중 오류 발생: {str(e)}")
        return False


def run_trading():
    global current_price, virtual_balance, initial_assets

    logger.info(f"===== {TICKER} 자동 매매 시작 =====")

    # 초기 설정
    logger.info(f"티커: {TICKER}")
    logger.info(f"차수별 주문 금액: {ORDER_AMOUNT:,}원")
    logger.info(f"가격 변동 기준: {PRICE_CHANGE}원")
    logger.info(f"최대 분할 차수: {MAX_GRID_COUNT}")
    logger.info(f"가격 확인 간격: {CHECK_INTERVAL}초")
    logger.info(f"미체결 주문 취소 시간: {CANCEL_TIMEOUT}초")
    logger.info(f"거래 수수료율: {FEE_RATE * 100:.3f}%")

    # 테스트 모드인 경우 초기 가상 잔고 설정
    if TEST_MODE:
        logger.info("테스트 모드 활성화됨")
        initial_assets = 0

        # 사용자 입력으로 초기 가상 잔고 설정 (선택사항)
        try:
            user_input = input("\n테스트 모드 초기 원화 잔고를 입력하세요 (기본값: 1,000,000원): ")
            if user_input.strip():
                virtual_balance["KRW"] = float(user_input.strip())

            initial_assets = virtual_balance["KRW"]

            user_input = input(f"\n테스트 모드 초기 {TICKER} 보유량을 입력하세요 (기본값: 0): ")
            if user_input.strip():
                virtual_balance["BTC"] = float(user_input.strip())

                if virtual_balance["BTC"] > 0:
                    user_input = input(f"\n테스트 모드 초기 {TICKER} 평균 매수가를 입력하세요 (기본값: 현재가): ")
                    if user_input.strip():
                        virtual_balance["BTC_AVG_PRICE"] = float(user_input.strip())
                    else:
                        # 현재가 조회
                        current_price_temp = get_current_price()
                        if current_price_temp:
                            virtual_balance["BTC_AVG_PRICE"] = current_price_temp
                        else:
                            virtual_balance["BTC_AVG_PRICE"] = 0
                            logger.error("현재가를 조회할 수 없어 평균 매수가를 0으로 설정합니다.")

                    # 초기 자산 계산에 BTC 포함
                    initial_assets += virtual_balance["BTC"] * virtual_balance["BTC_AVG_PRICE"]
        except ValueError:
            logger.error("올바른 숫자 형식이 아닙니다. 기본값을 사용합니다.")
            initial_assets = virtual_balance["KRW"]
        except KeyboardInterrupt:
            logger.info("사용자에 의해 프로그램이 중단되었습니다.")
            return

        logger.info(f"테스트 모드 초기 원화 잔고: {virtual_balance['KRW']:,.0f}원")
        logger.info(f"테스트 모드 초기 {TICKER} 보유량: {virtual_balance['BTC']:.8f} {TICKER}")
        if virtual_balance["BTC"] > 0:
            logger.info(f"테스트 모드 초기 {TICKER} 평균 매수가: {virtual_balance['BTC_AVG_PRICE']:,.2f}원")
            logger.info(f"테스트 모드 초기 {TICKER} 평가 금액: {virtual_balance['BTC'] * virtual_balance['BTC_AVG_PRICE']:,.0f}원")
        logger.info(f"테스트 모드 초기 총 자산: {initial_assets:,.0f}원")

    # 이전 상태 로드 시도
    loaded_previous_state = None  # load_trading_state()

    # 새 상태 생성 (이전 상태가 없는 경우)
    if not loaded_previous_state:
        # 잔고 확인
        balance = get_balance()
        if balance is None:
            logger.error("API 키 확인 필요 - 잔고를 가져올 수 없습니다.")
            return

        # 사용자 입력 받기
        try:
            price = get_current_price()
            if price:
                print(f"\n현재 {TICKER} 가격: {price:,.2f}원")

            user_input = input("\n기준 가격을 입력하세요 (엔터 시 현재 시장가 사용): ")
            if user_input.strip():
                try:
                    input_base_price = float(user_input.strip())
                    # 그리드 주문 생성 (사용자 입력 기준가)
                    if not create_grid_orders(input_base_price):
                        return
                except ValueError:
                    logger.error("올바른 숫자 형식이 아닙니다. 현재 시장가를 사용합니다.")
                    if not create_grid_orders():
                        return
            else:
                # 그리드 주문 생성 (현재 시장가)
                if not create_grid_orders():
                    return
        except KeyboardInterrupt:
            logger.info("사용자에 의해 프로그램이 중단되었습니다.")
            return
    else:
        # 잔고 확인
        get_balance()

    # 현재 가격 갱신
    current_price = get_current_price()
    if current_price is None:
        logger.error("현재 가격을 가져올 수 없습니다. 프로그램을 종료합니다.")
        return

    try:
        # 무한 루프 시작
        cycle_count = 0
        while True:
            cycle_count += 1
            logger.info(f"\n===== 사이클 #{cycle_count} =====")

            # 활성 주문 목록 표시
            logger.info(f"현재 활성 주문 수: {len(active_orders)}")

            # 주문 상태 확인
            check_orders()

            # 가격 확인 및 거래
            check_price_and_trade()

            # 주기적으로 거래 결과 저장 (10회마다)
            if cycle_count % 10 == 0:
                save_trading_results()

                # 테스트 모드에서 현재 가상 잔고 상태 표시
                if TEST_MODE:
                    logger.info("\n===== 테스트 모드 가상 잔고 상태 =====")
                    logger.info(f"원화 잔고: {virtual_balance['KRW']:,.0f}원")
                    logger.info(f"{TICKER} 보유량: {virtual_balance['BTC']:.8f} {TICKER}")
                    if virtual_balance["BTC"] > 0:
                        logger.info(f"{TICKER} 평균 매수가: {virtual_balance['BTC_AVG_PRICE']:,.2f}원")
                        logger.info(f"평가 금액: {virtual_balance['BTC'] * current_price:,.0f}원")

                    # 수익률 계산
                    if virtual_balance["BTC"] > 0 and virtual_balance["BTC_AVG_PRICE"] > 0:
                        profit_percentage = ((current_price / virtual_balance["BTC_AVG_PRICE"]) - 1) * 100
                        logger.info(f"현재 수익률: {profit_percentage:+.2f}%")

                    # 총자산 계산
                    total_assets = virtual_balance["KRW"] + (virtual_balance["BTC"] * current_price)
                    logger.info(f"총 자산: {total_assets:,.0f}원")

                    # 거래 통계
                    buy_count = sum(1 for trade in trade_history if trade['type'] == 'buy')
                    sell_count = sum(1 for trade in trade_history if trade['type'] == 'sell')
                    logger.info(f"현재까지 매수: {buy_count}회, 매도: {sell_count}회")

                    # 매수/매도 금액 및 수수료 합계
                    total_buy_amount = sum(trade['amount'] for trade in trade_history if trade['type'] == 'buy')
                    total_sell_amount = sum(trade['amount'] for trade in trade_history if trade['type'] == 'sell')
                    total_fee = sum(trade.get('fee', 0) for trade in trade_history)

                    if total_buy_amount > 0 or total_sell_amount > 0:
                        logger.info(f"총 매수 금액: {total_buy_amount:,.0f}원, 총 매도 금액: {total_sell_amount:,.0f}원")
                        logger.info(f"총 수수료: {total_fee:,.0f}원")

            # 대기
            logger.info(f"{CHECK_INTERVAL}초 대기...")
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        logger.info("\n사용자에 의해 거래가 중단되었습니다.")
    except Exception as e:
        logger.error(f"\n거래 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 최종 결과 출력 및 저장
        logger.info("===== 거래 종료 =====")
        get_balance()
        save_trading_results()

        # 테스트 모드인 경우 최종 거래 성과 분석
        if TEST_MODE:
            # 초기 자산 (프로그램 실행 시작 시)
            try:
                initial_assets
            except NameError:
                initial_assets = 1000000  # 기본값

            # 현재 자산
            final_assets = virtual_balance["KRW"] + (virtual_balance["BTC"] * current_price)

            # 수익 계산
            profit = final_assets - initial_assets
            profit_percentage = (profit / initial_assets) * 100 if initial_assets > 0 else 0

            logger.info("\n===== 테스트 모드 거래 결과 =====")
            logger.info(f"초기 자산: {initial_assets:,.0f}원")
            logger.info(f"최종 자산: {final_assets:,.0f}원")
            logger.info(f"총 수익: {profit:+,.0f}원 ({profit_percentage:+.2f}%)")

            # 거래 통계
            buy_count = sum(1 for trade in trade_history if trade['type'] == 'buy')
            sell_count = sum(1 for trade in trade_history if trade['type'] == 'sell')
            cancel_count = sum(1 for trade in trade_history if 'cancel' in trade['type'])

            logger.info(f"\n총 거래 횟수: {len(trade_history)}회")
            logger.info(f"매수: {buy_count}회, 매도: {sell_count}회, 취소: {cancel_count}회")

            # 매수/매도 금액 합계
            total_buy_amount = sum(trade['amount'] for trade in trade_history if trade['type'] == 'buy')
            total_sell_amount = sum(trade['amount'] for trade in trade_history if trade['type'] == 'sell')
            total_fee = sum(trade.get('fee', 0) for trade in trade_history)

            if total_buy_amount > 0:
                logger.info(f"총 매수 금액: {total_buy_amount:,.0f}원")
            if total_sell_amount > 0:
                logger.info(f"총 매도 금액: {total_sell_amount:,.0f}원")
                logger.info(f"순 거래 수익: {total_sell_amount - total_buy_amount:+,.0f}원")

            logger.info(f"총 수수료: {total_fee:,.0f}원")

        # 미체결 주문 취소 여부 확인
        if len(active_orders) > 0:
            logger.info(f"미체결 주문이 {len(active_orders)}개 있습니다.")
            logger.info("모든 미체결 주문을 취소합니다...")
            cancel_all_orders()

        # 거래 내역 요약
        buy_count = sum(1 for trade in trade_history if trade['type'] == 'buy')
        sell_count = sum(1 for trade in trade_history if trade['type'] == 'sell')
        cancel_count = sum(1 for trade in trade_history if 'cancel' in trade['type'])
        logger.info(f"\n총 거래 횟수: {len(trade_history)}회")
        logger.info(f"매수: {buy_count}회, 매도: {sell_count}회, 취소: {cancel_count}회")

# 메인 함수
if __name__ == "__main__":
    # API 키 확인
    if not ACCESS_KEY or not SECRET_KEY:
        logger.error("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        logger.info("예시 .env 파일 내용:")
        logger.info("UPBIT_ACCESS_KEY=your_access_key")
        logger.info("UPBIT_SECRET_KEY=your_secret_key")
    else:
        run_trading()