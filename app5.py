import os
import time
import logging
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import pyupbit

# .env 파일에서 환경변수 로드
load_dotenv()

# API 키 설정
ACCESS_KEY = os.environ.get("UPBIT_ACCESS_KEY")
SECRET_KEY = os.environ.get("UPBIT_SECRET_KEY")

# 거래 설정
TICKER = "KRW-USDT"  # 비트코인 티커
BASE_PRICE = None  # 초기 기준 가격 (시장 가격으로 설정됨)
PRICE_CHANGE = 2  # 가격 변동 기준 (원)
ORDER_AMOUNT = 10000  # 차수별 주문 금액 (원)
MAX_GRID_COUNT = 10  # 최대 분할 매수/매도 차수
CHECK_INTERVAL = 2  # 가격 확인 간격 (초)
CANCEL_TIMEOUT = 3600  # 미체결 주문 취소 시간 (초, 1시간)
TEST_MODE = True  # 테스트 모드 활성화 여부
FEE_RATE = 0.0005  # 거래 수수료 (0.05%)

# 전역 변수
current_price = 0  # 현재 가격
previous_price = None  # 이전 가격
price_oscillation_step = 0  # 가격 변동 단계 (0: 초기값, 1: +10원 상태, 2: -10원 상태)
grid_orders = []  # 그리드 주문 저장 리스트
trade_history = []  # 거래 내역 저장 리스트
active_orders = {}  # 활성 주문 관리: {uuid: {'grid_level': n, 'type': 'buy'/'sell', 'timestamp': datetime}}

# Upbit 객체 생성
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# 가상 잔고 정보를 저장할 전역 변수
virtual_balance = {
    "krw": 1000000,  # 초기 100만원
    "coin": 0,  # 초기 0 코인
    "coin_avg_price": 0  # 초기 평균 매수가 0원
}

# 로깅 설정
log_file = f"{TICKER.replace('KRW-', '').lower()}_grid_trade.log"

# 기본 로깅 설정 (콘솔 출력용)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 테스트 모드일 경우 기존 로그 파일 삭제
if TEST_MODE and os.path.exists(log_file):
    try:
        os.remove(log_file)
        logger.info(f"기존 로그 파일 삭제됨: {log_file}")
    except Exception as e:
        logger.error(f"로그 파일 삭제 중 오류 발생: {str(e)}")

# 파일 핸들러 추가
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# 로깅 설정 완료 로그
logger.info(f"로깅 설정 완료: {log_file}")

def get_current_price():
    """현재 가격 조회"""
    logger.info("get_current_price")
    global current_price, previous_price, price_oscillation_step

    try:
        if TEST_MODE:
            # 테스트 모드에서는 가상 가격 사용
            if 'current_price' not in globals() or current_price == 0:
                # 첫 호출 시 기본 시작 가격 설정 (사용자 입력 없이)
                current_price = pyupbit.get_current_price(TICKER)  # 초기 기본값은 티커코인의 현재가로 설정
                previous_price = current_price

                # 진동 단계 초기화 (0: 초기값, 1: +10원 상태, 2: -10원 상태)
                if 'price_oscillation_step' not in globals():
                    price_oscillation_step = 0

                ticker_price = current_price
            else:
                # 이전 가격 저장
                previous_price = current_price

                # 진동 패턴에 따라 가격 설정
                if price_oscillation_step == 0:
                    # 초기 상태에서 +2원 상태로
                    ticker_price = previous_price + 2
                    price_oscillation_step = 1
                elif price_oscillation_step == 1:
                    # +2원 상태에서 -2원 상태로 (초기값보다 -2원)
                    ticker_price = previous_price - 4  # +2원에서 -2원으로 (차이: -4원)
                    price_oscillation_step = 2
                else:  # price_oscillation_step == 2
                    # -2원 상태에서 다시 +2원 상태로
                    ticker_price = previous_price + 4  # -2원에서 +2원으로 (차이: +4원)
                    price_oscillation_step = 1

                current_price = ticker_price
        else:
            # 실제 API 호출
            ticker_price = pyupbit.get_current_price(TICKER)
            if 'previous_price' not in globals() or previous_price is None:
                previous_price = ticker_price
            if 'current_price' not in globals() or current_price == 0:
                current_price = ticker_price

        # 이전 가격과 비교하여 변동 계산
        if previous_price is None:
            # 처음 호출되는 경우
            price_change = 0
            change_percentage = 0
        else:
            # 가격 변동 계산
            price_change = ticker_price - previous_price
            change_percentage = (price_change / previous_price) * 100 if previous_price > 0 else 0

        # 변동 표시에 사용할 부호
        sign = "+" if price_change >= 0 else ""

        # 로그 출력
        price_msg = f"현재 {TICKER} 가격: {ticker_price:,.2f}원, ({sign}{change_percentage:.2f}%), {sign}{price_change:.2f}원 {'상승' if price_change >= 0 else '하락'}"
        logger.info(price_msg)
        print(price_msg)

        # 다음 비교를 위해 현재 가격을 이전 가격으로 저장
        if not TEST_MODE:
            previous_price = ticker_price
            current_price = ticker_price

        logger.info("/get_current_price\n")
        return ticker_price

    except Exception as e:
        logger.error(f"가격 조회 중 오류 발생: {str(e)}")

        # 오류 발생 시에도 테스트 모드에서는 가상 가격 반환
        if TEST_MODE and 'current_price' in globals() and current_price > 0:
            return current_price
        return None


def get_balance():
    """계좌 잔고 조회"""
    logger.info("get_balance")
    global virtual_balance

    try:
        if TEST_MODE:
            # 테스트 모드일 경우 가상 잔고 사용
            krw_balance = virtual_balance["krw"]
            btc_balance = virtual_balance["coin"]
            btc_avg_price = virtual_balance["coin_avg_price"] if btc_balance > 0 else 0

            logger.info(f"[테스트 모드] 보유 원화: {krw_balance:,.0f}원")
            logger.info(f"[테스트 모드] 보유 {TICKER}: {btc_balance:.8f} {TICKER}")

            if btc_balance > 0:
                logger.info(f"[테스트 모드] {TICKER} 평균 매수가: {btc_avg_price:,.2f}원")
                logger.info(f"[테스트 모드] 평가 금액: {btc_balance * current_price:,.0f}원")

            # 총자산 계산
            total_assets = krw_balance + (btc_balance * current_price)
            logger.info(f"[테스트 모드] 총 자산: {total_assets:,.0f}원")

            logger.info("/get_balance\n")
            return {
                "krw": krw_balance,
                "coin": btc_balance,
                "coin_avg_price": btc_avg_price,
                "total_assets": total_assets
            }
        else:
            # 실제 API 사용 (기존 코드)
            # 원화 잔고 조회
            krw_balance = upbit.get_balance("KRW")
            logger.info(f"보유 원화: {krw_balance:,.0f}원")

            # BTC 잔고 조회
            btc_balance = upbit.get_balance(TICKER)
            btc_avg_price = upbit.get_avg_buy_price(TICKER)
            logger.info(f"보유 {TICKER}: {btc_balance:.8f} {TICKER}")

            if btc_balance > 0:
                logger.info(f"{TICKER} 평균 매수가: {btc_avg_price:,.2f}원")
                logger.info(f"평가 금액: {btc_balance * current_price:,.0f}원")

            # 총자산 계산
            total_assets = krw_balance + (btc_balance * current_price)
            logger.info(f"총 자산: {total_assets:,.0f}원")

            logger.info("/get_balance\n")

            return {
                "krw": krw_balance,
                "coin": btc_balance,
                "coin_avg_price": btc_avg_price,
                "total_assets": total_assets
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

    logger.info("/create_grid_orders\n")
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
        logger.info("/buy_btc\n")
        return False

    # 이미 매수 주문이 있는 경우
    if grid['buy_order_id']:
        logger.info(f"레벨 {grid_level}은 이미 매수 주문이 진행 중입니다: {grid['buy_order_id']}")
        logger.info("/buy_btc\n")
        return False

    try:
        # 잔액 확인
        if TEST_MODE:
            krw_balance = virtual_balance["krw"]
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

        # UUID 생성 및 주문 정보 저장 (테스트 모드와 실제 모드 모두 동일하게 처리)
        if not TEST_MODE:
            try:
                # 실제 주문 실행
                order = upbit.buy_limit_order(TICKER, price, volume_without_fee)
                if order and 'uuid' in order:
                    uuid = order['uuid']
                else:
                    logger.error(f"매수 주문 실패: {order}")
                    return False
            except Exception as e:
                logger.error(f"매수 주문 API 호출 중 오류: {str(e)}")
                return False
        else:
            # 테스트용: 주문 UUID 생성
            uuid = f"buy-{grid_level}-{int(time.time())}"

        # 그리드 정보 업데이트 (테스트 모드와 실제 모드 모두 동일하게 처리)
        grid['buy_order_id'] = uuid
        grid['buy_order_time'] = datetime.now()

        # 활성 주문 목록에 추가 (테스트 모드와 실제 모드 모두 동일하게 처리)
        active_orders[uuid] = {
            'grid_level': grid_level,
            'type': 'buy',
            'price': price,
            'volume': volume_without_fee,
            'timestamp': datetime.now()
        }

        logger.info(f"매수 주문 완료: 주문번호 {uuid}")

        # 테스트 모드에서는 주문 금액만 가상 잔고에서 차감 (체결은 check_orders에서 처리)
        if TEST_MODE:
            # 주문 금액만큼 가상 잔고에서 예약 (실제 차감은 체결 시 처리)
            virtual_balance["krw"] -= order_amount
            logger.info(f"테스트 모드: 주문 금액 {order_amount:,}원 예약 (잔고: {virtual_balance['krw']:,}원)")

        logger.info("/buy_btc\n")
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
            btc_balance = virtual_balance["coin"]
        else:
            btc_balance = upbit.get_balance(TICKER)

        # 매도 수량 - 업비트는 매도 시 원화에서 수수료를 차감하므로 수량에 수수료 적용하지 않음
        volume = grid['volume']
        if btc_balance < volume:
            logger.warning(f"{TICKER} 보유량 부족: 매도 불가 (필요: {volume:.8f}, 보유: {btc_balance:.8f})")
            return False

        # 매도 가격
        price = grid['sell_price']

        if not TEST_MODE:
            try:
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
                    logger.info("/sell_btc\n")
                    return True
                else:
                    logger.error(f"매도 주문 실패: {order}")
                    return False
            except Exception as e:
                logger.error(f"매도 주문 API 호출 중 오류: {str(e)}")
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
            logger.info("/sell_btc\n")
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
                logger.info("/cancel_order\n")
                logger.info("/check_orders\n")
                return True
            else:
                logger.error(f"주문 취소 실패: {result}")
                logger.info("/cancel_order\n")
                logger.info("/check_orders\n")
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

                # 테스트 모드에서 취소 시 가상 잔고 복원
                if order_type == 'buy':
                    # 매수 주문 취소 시 원화 복원
                    order_amount = price * volume
                    virtual_balance["krw"] += order_amount
                    logger.info(f"테스트 모드: 취소된 매수 주문 금액 {order_amount:,}원 복원 (잔고: {virtual_balance['krw']:,}원)")

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
                logger.info("/cancel_order\n")
                logger.info("/check_orders\n")
            return True

    except Exception as e:
        logger.error(f"주문 취소 중 오류 발생: {str(e)}")
        return False


def check_orders():
    """주문 상태 확인 및 업데이트"""
    logger.info("check_orders")
    global virtual_balance

    # 현재 시간
    now = datetime.now()

    # 활성 주문 목록 복사 (for문에서 삭제 시 에러 방지)
    active_order_uuids = list(active_orders.keys())

    for uuid in active_order_uuids:
        order_info = active_orders[uuid]
        grid_level = order_info['grid_level']
        order_type = order_info['type']
        order_time = order_info['timestamp']
        price = order_info['price']
        volume = order_info['volume']

        # 해당 그리드 정보 가져오기
        grid = grid_orders[grid_level - 1]

        # 1. 주문 상태 확인
        try:
            if TEST_MODE:
                # 테스트 모드에서 주문 체결 시뮬레이션
                # 현재 가격과 주문 가격 비교하여 체결 여부 결정 또는 랜덤하게 체결 결정
                # 이 예시에서는 주문 후 일정 시간이 지나면 체결되는 것으로 처리
                time_diff = (now - order_time).total_seconds()
                # 1초 후 체결되는 것으로 가정 (실제 환경에 맞게 조정 가능)
                if time_diff > 1:
                    if order_type == 'buy':
                        # 매수 주문 체결 처리
                        grid['buy_filled'] = True
                        grid['status'] = 'bought'

                        # 수수료 계산
                        order_amount = price * volume
                        fee_amount = order_amount * FEE_RATE

                        # 테스트 모드에서 가상 잔고 업데이트
                        # 원화는 이미 차감된 상태, BTC 추가
                        virtual_balance["coin"] += volume * (1 - FEE_RATE)

                        # BTC 평균 매수가 업데이트
                        if virtual_balance["coin"] > 0:
                            # 평균 매수가 계산 로직
                            current_value = virtual_balance["coin_avg_price"] * (
                                        virtual_balance["coin"] - volume * (1 - FEE_RATE))
                            new_value = price * volume * (1 - FEE_RATE)
                            virtual_balance["coin_avg_price"] = (current_value + new_value) / virtual_balance["coin"] if \
                            virtual_balance["coin"] > 0 else 0

                        # 거래 내역 저장
                        trade = {
                            'type': 'buy',
                            'grid_level': grid_level,
                            'price': price,
                            'amount': order_amount,
                            'volume': volume * (1 - FEE_RATE),
                            'fee': fee_amount,
                            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                            'order_id': uuid
                        }
                        trade_history.append(trade)

                        logger.info(f"레벨 {grid_level}의 매수 주문이 체결되었습니다: {uuid}")
                        logger.info(
                            f"매수 완료: {volume * (1 - FEE_RATE):.8f} {TICKER} (가격: {price:,.2f}원, 수수료: {fee_amount:,.2f}원)")

                        # 활성 주문 목록에서 제거
                        del active_orders[uuid]

                        # 매수 주문이 체결되면 즉시 매도 주문 실행
                        logger.info(f"레벨 {grid_level}의 매수 주문이 체결되었습니다. 매도 주문 실행")
                        sell_btc(grid_level)

                    else:  # sell
                        # 매도 주문 체결 처리
                        grid['sell_filled'] = True
                        grid['status'] = 'sold'

                        # 수수료 계산
                        sell_amount = volume * price  # 총 매도 금액
                        fee_amount = sell_amount * FEE_RATE  # 수수료 금액
                        amount = sell_amount - fee_amount  # 수수료를 제외한 실제 매도 금액

                        # 테스트 모드에서 가상 잔고 업데이트
                        virtual_balance["krw"] += amount  # 수수료 제외한 금액 추가
                        virtual_balance["coin"] -= volume  # 전체 수량 차감

                        # BTC가 매우 작은 값이거나 0이 되면 완전히 초기화
                        if virtual_balance["coin"] < 0.00000001:
                            virtual_balance["coin"] = 0
                            virtual_balance["coin_avg_price"] = 0

                        # 수익 계산
                        profit = amount - grid['order_amount']
                        profit_percentage = (profit / grid['order_amount']) * 100

                        # 거래 내역 저장
                        trade = {
                            'type': 'sell',
                            'grid_level': grid_level,
                            'price': price,
                            'amount': amount,
                            'volume': volume,
                            'fee': fee_amount,
                            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                            'order_id': uuid
                        }
                        trade_history.append(trade)

                        logger.info(f"레벨 {grid_level}의 매도 주문이 체결되었습니다: {uuid}")
                        logger.info(
                            f"매도 완료: {volume:.8f} {TICKER} (가격: {price:,.2f}원, 금액: {amount:,.0f}원, 수수료: {fee_amount:,.2f}원)")
                        logger.info(f"레벨 {grid_level} 수익: {profit:+,.0f}원 ({profit_percentage:+.2f}%)")

                        # 그리드 초기화 (재사용 가능하도록)
                        grid['buy_filled'] = False
                        grid['sell_filled'] = False
                        grid['status'] = 'waiting'
                        grid['buy_order_id'] = None
                        grid['sell_order_id'] = None
                        grid['buy_order_time'] = None
                        grid['sell_order_time'] = None

                        # 활성 주문 목록에서 제거
                        del active_orders[uuid]

                    # 잔고 출력
                    get_balance()
                
            else:
                order = upbit.get_order(uuid)

                # 체결된 경우
                if order['state'] == 'done':
                    if order_type == 'buy':
                        grid['buy_filled'] = True
                        grid['status'] = 'bought'

                        # 매수 주문이 체결되면 즉시 매도 주문 실행
                        logger.info(f"레벨 {grid_level}의 매수 주문이 체결되었습니다. 매도 주문 실행")
                        sell_btc(grid_level)

                    else:  # sell
                        grid['sell_filled'] = True
                        grid['status'] = 'sold'

                        # 그리드 초기화 (매도 완료 후)
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

    logger.info("/check_orders\n")


def check_price_and_trade():
    """현재 가격을 확인하고 모든 그리드 주문에 대해 거래 실행"""
    logger.info("check_price_and_trade")
    global current_price

    # 현재 가격 업데이트
    current_price = get_current_price()
    if current_price is None:
        logger.error("가격 조회 실패")
        logger.info("/check_price_and_trade\n")
        return

    # 모든 그리드 주문 확인
    for grid in grid_orders:
        level = grid['level']

        # 매수 조건: 현재 가격이 매수가 이하이고 아직 매수되지 않은 경우 (주문 없는 경우)
        if current_price <= grid['buy_price'] and not grid['buy_filled'] and not grid['buy_order_id']:
            logger.info(f"레벨 {level} 매수 조건 충족: 현재가({current_price:,.2f}원) <= 매수가({grid['buy_price']:,.2f}원)")
            buy_success = buy_btc(level)

            # 매수 성공 로그만 남기고, 실제 모드와 테스트 모드 모두 check_orders에서 체결 처리
            if buy_success:
                logger.info(f"레벨 {level} 매수 주문 완료. 체결 후 자동으로 매도 주문이 실행됩니다.")

        # 매도 조건: 현재 가격이 매도가 이상이고 이미 매수되었지만 아직 매도되지 않은 경우 (주문 없는 경우)
        elif current_price >= grid['sell_price'] and grid['buy_filled'] and not grid['sell_filled'] and not grid[
            'sell_order_id']:
            logger.info(f"레벨 {level} 매도 조건 충족: 현재가({current_price:,.2f}원) >= 매도가({grid['sell_price']:,.2f}원)")
            sell_btc(level)

    logger.info("/check_price_and_trade\n")


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
    logger.info("/cancel_all_orders\n")
    return canceled_count


def display_virtual_balance():
    """테스트 모드에서 현재 가상 잔고 상태를 표시하는 함수"""
    logger.info("display_virtual_balance")
    logger.info("===== 테스트 모드 가상 잔고 상태 =====")
    logger.info(f"원화 잔고: {virtual_balance['krw']:,.0f}원")
    logger.info(f"{TICKER} 보유량: {virtual_balance['coin']:.8f} {TICKER}")
    if virtual_balance["coin"] > 0:
        logger.info(f"{TICKER} 평균 매수가: {virtual_balance['coin_avg_price']:,.2f}원")
        logger.info(f"평가 금액: {virtual_balance['coin'] * current_price:,.0f}원")

    # 수익률 계산
    if virtual_balance["coin"] > 0 and virtual_balance["coin_avg_price"] > 0:
        profit_percentage = ((current_price / virtual_balance["coin_avg_price"]) - 1) * 100
        logger.info(f"현재 수익률: {profit_percentage:+.2f}%")

    # 총자산 계산
    total_assets = virtual_balance["krw"] + (virtual_balance["coin"] * current_price)
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
    
    logger.info("/display_virtual_balance\n")


def display_final_trading_results():
    """테스트 모드에서 최종 거래 결과를 표시하는 함수"""
    logger.info("display_final_trading_results")
    global initial_assets
    
    # 초기 자산 (프로그램 실행 시작 시)
    try:
        initial_assets
    except NameError:
        initial_assets = 1000000  # 기본값

    # 현재 자산
    final_assets = virtual_balance["krw"] + (virtual_balance["coin"] * current_price)

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
    logger.info("/display_final_trading_results\n")


def run_trading():
    logger.info("run_trading")
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

        # 사용자 입력으로 초기 가상 잔고 설정
        initial_assets = virtual_balance["krw"]  # 초기 원화 잔고 (기본값: 1,000,000원)
        logger.info(f"초기 원화 잔고: {virtual_balance['krw']:,.0f}원")
        logger.info(f"초기 {TICKER} 보유량: {virtual_balance['coin']:.8f} {TICKER}")
        logger.info(f"초기 총 자산: {initial_assets:,.0f}원")


    try:
        ticker_price = get_current_price()  #

        input_base_price = ticker_price  # 기준 가격을 현재 시장가 사용로 설정
        # 그리드 주문 생성 (사용자 입력 기준가)
        if not create_grid_orders(input_base_price):
            return
    except Exception as e:
        logger.error(f"\ncreate_grid_orders 오류 발생: {str(e)}")
        return

    # 현재 가격 갱신
    current_price = get_current_price()
    if current_price is None:
        logger.error("현재 가격을 가져올 수 없습니다. 프로그램을 종료합니다.")
        return

    try:
        # 무한 루프 시작
        logger.info(f"===== 무한 루프 시작(run_trading) =====")
        cycle_count = 0
        while True:
            cycle_count += 1
            logger.info(f"===== 사이클 #{cycle_count} =====")

            # 활성 주문 목록 표시
            logger.info(f"현재 활성 주문 수: {len(active_orders)}")

            # 주문 상태 확인
            check_orders()

            # 가격 확인 및 거래
            check_price_and_trade()

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

        # 테스트 모드인 경우 최종 거래 성과 분석
        if TEST_MODE:
            display_final_trading_results()

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

    logger.info("/run_trading\n")


# 메인 함수
if __name__ == "__main__":
    logger.info("main")
    # API 키 확인
    if not ACCESS_KEY or not SECRET_KEY:
        logger.error("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
    else:
        run_trading()
    logger.info("/main\n")