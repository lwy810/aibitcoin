# 기본 라이브러리 임포트
import os  # 운영체제 관련 기능 사용
import time  # 시간 지연 및 시간 관련 기능
import logging  # 로깅 기능
import numpy as np  # 수치 계산
from datetime import datetime  # 날짜/시간 처리
from dotenv import load_dotenv  # 환경변수 관리
import pybithumb # 빗썸 API 연동
import requests  # HTTP 요청
import sys  # 시스템 관련 기능

# Windows 환경에서만 winsound 모듈 import (소리 알림용)
if sys.platform == 'win32':
    import winsound

# .env 파일에서 환경변수 로드
load_dotenv()

# API 키 설정
ACCESS_KEY = os.environ.get("BITHUMB_ACCESS_KEY")
SECRET_KEY = os.environ.get("BITHUMB_SECRET_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# 실거래 설정
COIN_SYMBOL = "XRP"  # 거래할 코인 심볼 (예: "BTC", "XRP")
PAYMENT_CURRENCY = "KRW" # 결제 통화 (빗썸 기본값)
TICKER = f"{COIN_SYMBOL}_{PAYMENT_CURRENCY}" # 내부 로깅 및 파일명용 티커 표현

BASE_PRICE = None  # 기준 가격 (자동 설정됨)
PRICE_CHANGE = 4  # 가격 변동 기준(단위:원)
OFFSET_GRID = 4  # 기준가로부터의 구간 오프셋(단위:구간 0<= N <=10)
ORDER_AMOUNT = 5000  # 주문당 금액 (단위:원, 최소주문금액 5000원)
MAX_GRID_COUNT = 10  # 최대 그리드 수(단위:구간 1<= N <=100)
CHECK_INTERVAL = 10  # 가격 체크 간격 (단위:초)
    
FEE_RATE = 0.0004  # 빗썸 거래 수수료 (예: 0.05%)
DISCORD_LOGGING = False  # 디스코드 로깅 비활성화

# 전역 변수
current_price = 0  # 현재 코인 가격
previous_price = None  # 이전 가격
grid_orders = []  # 그리드 주문 목록
trade_history = []  # 거래 내역

class DiscordLogger:
    """디스코드로 로그를 전송하는 전용 로거"""
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)

    def send(self, message, level="INFO"):
        """디스코드로 메시지 전송
        level: INFO(초록색), WARNING(노란색), ERROR(빨간색)
        """
        if not self.enabled:
            return

        try:
            color = {
                'INFO': 0x00ff00,    # 초록색
                'WARNING': 0xffff00,  # 노란색
                'ERROR': 0xff0000,    # 빨간색
                'CRITICAL': 0xff0000  # 빨간색
            }.get(level, 0x808080)  # 기본 회색

            kst_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            payload = {
                "embeds": [{
                    "title": f"[{level}]",
                    "description": f"{message}\n\n{kst_time} (KST)",
                    "color": color
                }]
            }
            requests.post(self.webhook_url, json=payload)
        except Exception as e:
            print(f"Discord 로그 전송 중 오류 발생: {str(e)}")

discord_logger = DiscordLogger(DISCORD_WEBHOOK_URL)

log_file = f"{COIN_SYMBOL.lower()}_grid_trade_bithumb.log" # 로그 파일명 변경
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', 
                                          datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(file_handler)
logger.info("로깅 설정 완료")

# Bithumb 객체 생성
try:
    bithumb_api = pybithumb.Bithumb(ACCESS_KEY, SECRET_KEY)
    logger.info("빗썸 API 객체 생성 성공")
except Exception as e:
    logger.error(f"빗썸 API 객체 생성 실패: {e}")
    bithumb_api = None # API 객체 생성 실패 시 None으로 초기화

def get_current_price():
    """현재 가격 조회"""
    logger.info("get_current_price (Bithumb)")
    global current_price, previous_price

    try:
        ticker_price_float = pybithumb.get_current_price(COIN_SYMBOL)
        if ticker_price_float is None:
            logger.error(f"{COIN_SYMBOL} 가격 조회 실패 (None 반환)")
            return None
        
        ticker_price = float(ticker_price_float)

        if 'previous_price' not in globals() or previous_price is None:
            previous_price = ticker_price
        if 'current_price' not in globals() or current_price == 0:
            current_price = ticker_price

        price_change_val = 0
        change_percentage = 0
        if previous_price is not None and previous_price > 0 :
            price_change_val = ticker_price - previous_price
            change_percentage = (price_change_val / previous_price) * 100
        
        sign = "+" if price_change_val >= 0 else ""
        base_price_str = f"{BASE_PRICE:,.2f}원" if BASE_PRICE is not None else "미설정"
        
        grid_level_info = "미설정"
        if BASE_PRICE is not None and grid_orders:
            for grid_item in grid_orders:
                if grid_item['buy_price_target'] >= ticker_price > grid_item['buy_price_target'] - PRICE_CHANGE:
                    grid_level_info = f"구간 {grid_item['level']} 매수 목표가({grid_item['buy_price_target']:,.2f}원) 근접"
                    break
            else:
                if grid_orders and ticker_price > grid_orders[0]['buy_price_target']:
                  grid_level_info = "최상단 매수 구간 초과"
                elif grid_orders and ticker_price < grid_orders[-1]['buy_price_target'] - PRICE_CHANGE :
                  grid_level_info = "최하단 매수 구간 미만"

        price_msg = f"현재 {COIN_SYMBOL} 가격: {ticker_price:,.2f}원, 기준가: {base_price_str}, {grid_level_info}, ({sign}{change_percentage:.2f}%), {sign}{price_change_val:.2f}원 {'상승' if price_change_val >= 0 else '하락'}"
        logger.info(price_msg)
        
        previous_price = ticker_price
        current_price = ticker_price

        logger.info("/get_current_price (Bithumb)\n")
        return ticker_price

    except Exception as e:
        logger.error(f"가격 조회 중 오류 발생: {str(e)}")
        return None


def get_balance():
    """계좌 잔고 조회"""
    logger.info("get_balance (Bithumb)")

    try:
        balance_krw_info = bithumb_api.get_balance(PAYMENT_CURRENCY) # (total, used, available, xcoin_created)
        krw_balance = float(balance_krw_info[2]) if balance_krw_info else 0.0

        balance_coin_info = bithumb_api.get_balance(COIN_SYMBOL) # (total, used, available, avg_buy_price)
        coin_balance = float(balance_coin_info[2]) if balance_coin_info else 0.0
        coin_avg_price = float(balance_coin_info[3]) if balance_coin_info and coin_balance > 0 else 0.0
        
        logger.info(f"보유 {PAYMENT_CURRENCY}: {krw_balance:,.0f}원")
        logger.info(f"보유 {COIN_SYMBOL}: {coin_balance:.8f} {COIN_SYMBOL}")

        current_coin_value = 0
        if coin_balance > 0 and current_price > 0 :
            current_coin_value = coin_balance * current_price
            total_investment = coin_balance * coin_avg_price if coin_avg_price > 0 else 0 # 평단가 0인 경우 방지
            profit = current_coin_value - total_investment
            profit_percentage = (profit / total_investment) * 100 if total_investment > 0 else 0
            
            logger.info(f"{COIN_SYMBOL} 평균 매수가: {coin_avg_price:,.2f}원")
            logger.info(f"평가 금액: {current_coin_value:,.0f}원")
            logger.info(f"수익금: {profit:+,.0f}원 ({profit_percentage:+.2f}%)")

        total_assets = krw_balance + current_coin_value
        logger.info(f"총 자산: {total_assets:,.0f}원")

        logger.info("/get_balance (Bithumb)\n")
        return {
            "krw": krw_balance,
            "coin": coin_balance,
            "coin_avg_price": coin_avg_price,
            "total_assets": total_assets
        }
    except Exception as e:
        logger.error(f"잔고 조회 중 오류 발생: {str(e)}")
        return None


def create_grid_orders(input_base_price=None):
    """분할 매수/매도 그리드 주문 생성"""
    logger.info("create_grid_orders (Bithumb)")
    global grid_orders, BASE_PRICE

    current_market_price = get_current_price()
    if current_market_price is None:
        logger.error("현재 가격을 가져올 수 없습니다. 그리드 생성을 중단합니다.")
        return False

    if input_base_price is not None:
        BASE_PRICE = input_base_price
        logger.info(f"사용자 지정 기준 가격: {BASE_PRICE:,.2f}원")
    else:
        BASE_PRICE = current_market_price + (PRICE_CHANGE * OFFSET_GRID)
        logger.info(f"현재 시장 가격: {current_market_price:,.2f}원")
        logger.info(f"기준 가격 설정 (현재가 + {OFFSET_GRID}구간): {BASE_PRICE:,.2f}원")

    grid_orders = []
    for i in range(MAX_GRID_COUNT):
        buy_target_price = BASE_PRICE - (i * PRICE_CHANGE)
        sell_target_price = buy_target_price + PRICE_CHANGE

        grid = {
            'level': i + 1,
            'buy_price_target': buy_target_price,
            'sell_price_target': sell_target_price,
            'order_krw_amount': ORDER_AMOUNT,
            'is_bought': False,
            'actual_bought_volume': 0.0,
            'actual_buy_fill_price': 0.0
        }
        grid_orders.append(grid)

    logger.info(f"총 {len(grid_orders)}개의 그리드 주문 설정 생성됨 ({COIN_SYMBOL})")
    logger.info(f"매수/매도 가격 간격: {PRICE_CHANGE}원")
    logger.info(f"주문당 KRW 금액: {ORDER_AMOUNT:,}원")
    for grid in grid_orders:
        logger.info(
            f"{grid['level']}차: 매수 목표가 {grid['buy_price_target']:,.2f}원, 매도 목표가 {grid['sell_price_target']:,.2f}원")

    logger.info("/create_grid_orders (Bithumb)\n")
    return True


def play_sound(sound_type):
    """거래 알림음 재생"""
    try:
        if sys.platform != 'win32':
            logger.info(f"사운드 재생: {sound_type} (Windows 환경에서만 지원)")
            return
        if sound_type == 'buy':
            winsound.PlaySound('buy.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif sound_type == 'sell':
            winsound.PlaySound('sell.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        logger.error(f"알림음 재생 중 오류 발생: {str(e)}")
        logger.info("알림음 파일(buy.wav, sell.wav)이 현재 디렉토리에 있는지 확인하세요.")


def buy_coin(grid_level):
    """지정된 그리드 레벨에서 코인 시장가 매수 (빗썸)"""
    logger.info(f"buy_coin (Level {grid_level}, Bithumb)")
    global current_price
    if bithumb_api is None:
        logger.error("Bithumb API 객체가 초기화되지 않았습니다. 매수 불가.")
        return False

    grid = grid_orders[grid_level - 1]

    if grid['is_bought']:
        logger.info(f"레벨 {grid_level}은 이미 매수 상태입니다.")
        logger.info("/buy_coin (Bithumb)\n")
        return False

    try:
        balance_info = get_balance() # 현재 잔고 확인
        if balance_info is None:
             logger.error("원화 잔고 조회 실패 (buy_coin, Bithumb)")
             return False
        krw_balance = balance_info["krw"]
        
        order_krw_amount_for_grid = grid['order_krw_amount'] # 그리드에 설정된 주문 금액
        if krw_balance < order_krw_amount_for_grid:
            logger.warning(f"잔액 부족: 매수 불가 (필요: {order_krw_amount_for_grid:,}원, 보유: {krw_balance:,}원)")
            return False

        # 시장가 매수 시점의 가격 사용
        actual_fill_price = current_price 
        if actual_fill_price <= 0:
            logger.error(f"유효하지 않은 시장 가격 ({actual_fill_price})으로 매수 불가.")
            return False

        # 빗썸은 수량 기준 주문. 주문 금액으로 수량 계산 (수수료 고려)
        # 실제 코인 구매에 사용될 KRW (수수료 제외)
        krw_for_coin_purchase_net = order_krw_amount_for_grid * (1 - FEE_RATE)
        units_to_buy = krw_for_coin_purchase_net / actual_fill_price
        
        # 빗썸 최소 주문 수량 등 제약조건 확인 필요 (pybithumb에서 처리하거나, 여기서 추가 검증)
        # 예: XRP의 최소 주문 수량 등 확인

        logger.info(f"시장가 매수 시도: {units_to_buy:.8f} {COIN_SYMBOL} @ 약 {actual_fill_price:,.2f}원 (총 주문액 기준: {order_krw_amount_for_grid:,.0f}원)")
        
        # 빗썸 API 시장가 매수 호출
        # buy_market_order(ticker, unit)
        order_result = None # 초기화
        try:
            order_result = bithumb_api.buy_market_order(COIN_SYMBOL, units_to_buy)
            # order_result는 주문 성공 시 (status, order_id, ...) 형태의 튜플 또는 주문 ID 등을 반환할 수 있음
            # pybithumb 문서에 따르면, 성공 시 order_id를 포함한 튜플 반환 가능
            # 예: ('placed', '1607390097370790', 'XRP', 'KRW')
            # 또는 특정 경우 바로 체결 결과 리스트를 반환할 수도 있음. 여기서는 order_id가 온다고 가정.
            if isinstance(order_result, tuple) and len(order_result) > 1:
                order_id = order_result[1]
                logger.info(f"매수 주문 요청 성공: Order ID {order_id} ({COIN_SYMBOL})")
            elif isinstance(order_result, dict) and "order_id" in order_result: # 다른 반환 형식 가능성
                order_id = order_result["order_id"]
                logger.info(f"매수 주문 요청 성공: Order ID {order_id} ({COIN_SYMBOL})")
            else: # 단순 ID 반환 또는 기타 형식
                logger.info(f"매수 주문 요청 결과: {order_result} ({COIN_SYMBOL})")
                # order_id를 특정할 수 없는 경우, 이후 체결 확인이 어려울 수 있음
                # 여기서는 주문이 성공적으로 접수되었다고 가정하고 진행

        except Exception as e:
            logger.error(f"빗썸 매수 주문 API 호출 중 오류: {str(e)}")
            return False

        # 실제 체결 정보는 주문 직후 알 수 없을 수 있음.
        # 여기서는 주문 시점의 가격을 체결가로, 계산된 수량을 체결량으로 간주.
        # 수수료는 총 주문액 기준
        actual_bought_volume = units_to_buy 
        fee_paid_krw = order_krw_amount_for_grid * FEE_RATE


        grid['is_bought'] = True
        grid['actual_bought_volume'] = actual_bought_volume 
        grid['actual_buy_fill_price'] = actual_fill_price 

        trade = {
            'type': 'buy',
            'grid_level': grid_level,
            'price': actual_fill_price, # 주문 시점의 가격
            'amount': order_krw_amount_for_grid, # 총 주문 금액 (KRW)
            'volume': actual_bought_volume, # 계산된 매수 코인 양
            'fee': fee_paid_krw, # 계산된 수수료 (KRW)
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        trade_history.append(trade)

        logger.info(f"매수 주문 접수 (L{grid_level}): {actual_bought_volume:.8f} {COIN_SYMBOL} @ {actual_fill_price:,.2f}원 (주문액: {order_krw_amount_for_grid:,.0f}원, 예상수수료: {fee_paid_krw:,.2f}원)")
        if DISCORD_LOGGING:
            discord_logger.send(f"매수 주문 접수 (L{grid_level}): {actual_bought_volume:.8f} {COIN_SYMBOL} @ {actual_fill_price:,.2f}원 (주문액: {order_krw_amount_for_grid:,.0f}원, 예상수수료: {fee_paid_krw:,.2f}원)", "INFO")
        
        play_sound('buy')
        time.sleep(2) # 체결 및 잔고 반영 대기
        get_balance() # 거래 후 잔고 업데이트 및 로깅
        logger.info("/buy_coin (Bithumb)\n")
        return True

    except Exception as e:
        logger.error(f"매수 중 오류 발생 (L{grid_level}): {str(e)}")
        return False


def sell_coin(grid_level):
    """지정된 그리드 레벨에서 코인 시장가 매도 (빗썸)"""
    logger.info(f"sell_coin (Level {grid_level}, Bithumb)")
    global current_price
    if bithumb_api is None:
        logger.error("Bithumb API 객체가 초기화되지 않았습니다. 매도 불가.")
        return False

    grid = grid_orders[grid_level - 1]

    if not grid['is_bought']:
        logger.info(f"레벨 {grid_level}은 매수 상태가 아니므로 매도 불가.")
        return False

    volume_to_sell = grid['actual_bought_volume'] # 이전에 매수된 것으로 기록된 수량
    if volume_to_sell <= 0:
        logger.warning(f"레벨 {grid_level} 매도할 코인 수량 없음 ({volume_to_sell:.8f}).")
        return False
        
    try:
        balance_info = get_balance()
        if balance_info is None:
            logger.error("코인 잔고 조회 실패 (sell_coin, Bithumb)")
            return False
        coin_balance = balance_info["coin"]

        if coin_balance < volume_to_sell:
            logger.warning(f"{COIN_SYMBOL} 보유량 부족: 매도 불가 (필요: {volume_to_sell:.8f}, 보유: {coin_balance:.8f}). 실제 매도 가능 수량으로 조정 시도.")
            volume_to_sell = coin_balance # 실제 보유량만큼만 매도 시도 (부분 매도)
            if volume_to_sell <=0:
                logger.error("실제 매도 가능 수량 없음.")
                return False


        actual_fill_price = current_price # 시장가 매도 시점의 가격 사용
        if actual_fill_price <= 0:
            logger.error(f"유효하지 않은 시장 가격 ({actual_fill_price})으로 매도 불가.")
            return False

        logger.info(f"시장가 매도 시도: {volume_to_sell:.8f} {COIN_SYMBOL} @ 약 {actual_fill_price:,.2f}원")
        
        order_result = None
        try:
            order_result = bithumb_api.sell_market_order(COIN_SYMBOL, volume_to_sell)
            if isinstance(order_result, tuple) and len(order_result) > 1:
                order_id = order_result[1]
                logger.info(f"매도 주문 요청 성공: Order ID {order_id} ({COIN_SYMBOL})")
            elif isinstance(order_result, dict) and "order_id" in order_result:
                order_id = order_result["order_id"]
                logger.info(f"매도 주문 요청 성공: Order ID {order_id} ({COIN_SYMBOL})")
            else:
                logger.info(f"매도 주문 요청 결과: {order_result} ({COIN_SYMBOL})")
        
        except Exception as e:
            logger.error(f"빗썸 매도 주문 API 호출 중 오류: {str(e)}")
            return False
        
        # 매도 후 실제 받은 KRW 계산
        gross_sell_krw = volume_to_sell * actual_fill_price
        fee_paid_krw = gross_sell_krw * FEE_RATE
        net_sell_krw_received = gross_sell_krw * (1 - FEE_RATE)

        profit_for_this_trade = net_sell_krw_received - grid['order_krw_amount'] # 해당 그리드 초기 투자금 대비 수익
        profit_percentage = (profit_for_this_trade / grid['order_krw_amount']) * 100 if grid['order_krw_amount'] > 0 else 0
        
        # 그리드 상태 초기화
        grid['is_bought'] = False
        grid['actual_bought_volume'] = 0.0 
        grid['actual_buy_fill_price'] = 0.0
        
        trade = {
            'type': 'sell',
            'grid_level': grid_level,
            'price': actual_fill_price, # 주문 시점 가격
            'amount': net_sell_krw_received, # 실제 받은 KRW (예상, 수수료 제외)
            'volume': volume_to_sell,
            'fee': fee_paid_krw, # 예상 수수료 (KRW)
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        trade_history.append(trade)

        logger.info(f"매도 주문 접수 (L{grid_level}): {volume_to_sell:.8f} {COIN_SYMBOL} @ {actual_fill_price:,.2f}원 (예상실현금액: {net_sell_krw_received:,.0f}원, 예상수수료: {fee_paid_krw:,.2f}원)")
        logger.info(f"레벨 {grid_level} 거래 예상 수익: {profit_for_this_trade:+,.0f}원 ({profit_percentage:+.2f}%)")
        if DISCORD_LOGGING:
            discord_logger.send(f"매도 주문 접수 (L{grid_level}): {volume_to_sell:.8f} {COIN_SYMBOL} @ {actual_fill_price:,.2f}원 (예상실현금액: {net_sell_krw_received:,.0f}원)\n예상수익: {profit_for_this_trade:+,.0f}원 ({profit_percentage:+.2f}%)", "INFO")
        
        play_sound('sell')
        time.sleep(2) # 체결 및 잔고 반영 대기
        get_balance() # 거래 후 잔고 업데이트 및 로깅
        logger.info("/sell_coin (Bithumb)\n")
        return True

    except Exception as e:
        logger.error(f"빗썸 매도 중 오류 발생 (L{grid_level}): {str(e)}")
        return False


def check_price_and_trade():
    """현재 가격을 확인하고 모든 그리드 주문에 대해 거래 실행"""
    logger.info("check_price_and_trade (Bithumb)")
    global current_price 

    if current_price is None or current_price <= 0: 
        logger.error("유효하지 않은 현재 가격으로 거래 로직을 실행할 수 없습니다.")
        logger.info("/check_price_and_trade (Bithumb)\n")
        return

    for i, grid in enumerate(grid_orders): 
        level = grid['level'] 
        
        if not grid['is_bought'] and current_price <= grid['buy_price_target']:
            logger.info(f"레벨 {level} 매수 조건 충족: 현재가({current_price:,.2f}원) <= 매수 목표가({grid['buy_price_target']:,.2f}원)")
            buy_coin(level) 
            time.sleep(1) 

        elif grid['is_bought'] and current_price >= grid['sell_price_target']:
            logger.info(f"레벨 {level} 매도 조건 충족: 현재가({current_price:,.2f}원) >= 매도 목표가({grid['sell_price_target']:,.2f}원)")
            sell_coin(level) 
            time.sleep(1) 
            
    logger.info("/check_price_and_trade (Bithumb)\n")


def run_trading():
    logger.info("run_trading (Bithumb)")
    global current_price, DISCORD_LOGGING

    if bithumb_api is None:
        logger.error("Bithumb API가 초기화되지 않아 거래를 시작할 수 없습니다.")
        return

    logger.info(f"===== {COIN_SYMBOL} 자동 매매 시작 (Bithumb) =====")
    logger.info(f"코인: {COIN_SYMBOL}")
    logger.info(f"차수별 주문 금액: {ORDER_AMOUNT:,}원")
    logger.info(f"가격 변동 기준: {PRICE_CHANGE}원")
    logger.info(f"최대 분할 차수: {MAX_GRID_COUNT}")
    logger.info(f"가격 확인 간격: {CHECK_INTERVAL}초")
    logger.info(f"거래 수수료율: {FEE_RATE * 100:.3f}%")
    logger.info(f"Discord 로깅: {'활성화' if DISCORD_LOGGING else '비활성화'}")

    try:
        logger.info("실거래 모드 활성화됨 (Bithumb)")
        temp_current_price = get_current_price() 
        if temp_current_price is None:
            logger.error("프로그램 시작 시 현재 가격 조회 실패. 종료합니다.")
            return

        input_base_price_for_grid = temp_current_price + (PRICE_CHANGE * OFFSET_GRID)
        logger.info(f"현재 시장 가격 ({COIN_SYMBOL}): {temp_current_price:,.2f}원")
        logger.info(f"그리드 기준 가격 자동 설정 (현재가 + {OFFSET_GRID}구간): {input_base_price_for_grid:,.2f}원")

        if not create_grid_orders(input_base_price_for_grid):
            logger.error("그리드 주문 생성 실패. 프로그램을 종료합니다.")
            return
        
        initial_balance = get_balance()
        if initial_balance:
            logger.info(f"초기 자산: {initial_balance.get('total_assets', 0):,.0f}원 (KRW: {initial_balance.get('krw',0):,.0f}, {COIN_SYMBOL}: {initial_balance.get('coin',0):.8f})")


        logger.info(f"===== 매매 루프 시작 ({COIN_SYMBOL}) =====")
        cycle_count = 0
        while True:
            cycle_count = (cycle_count % 10000) + 1
            logger.info(f"===== 사이클 #{cycle_count} ({COIN_SYMBOL}) =====")
            
            current_price = get_current_price() 
            if current_price is None:
                logger.warning("현재 가격 조회 실패. 다음 사이클까지 대기합니다.")
                time.sleep(CHECK_INTERVAL)
                continue

            check_price_and_trade() 
            
            logger.info(f"{CHECK_INTERVAL}초 대기...")
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        logger.info("\n사용자에 의해 거래가 중단되었습니다.")
    except Exception as e:
        logger.error(f"\n거래 중 예외 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"===== 거래 로직 종료 ({COIN_SYMBOL}) =====")
        if current_price is None or current_price == 0: 
            current_price = get_current_price() or 0 
        
        final_balance = get_balance() 
        if final_balance:
             logger.info(f"최종 자산: {final_balance.get('total_assets', 0):,.0f}원 (KRW: {final_balance.get('krw',0):,.0f}, {COIN_SYMBOL}: {final_balance.get('coin',0):.8f})")


        buy_count = sum(1 for trade in trade_history if trade['type'] == 'buy')
        sell_count = sum(1 for trade in trade_history if trade['type'] == 'sell')
        logger.info(f"\n최종 거래 요약: 총 {buy_count + sell_count}회 체결 (매수: {buy_count}회, 매도: {sell_count}회)")

    logger.info("/run_trading (Bithumb)\n")


if __name__ == "__main__":
    logger.info("main (Bithumb version)")
    if not ACCESS_KEY or not SECRET_KEY: 
        logger.error("BITHUMB_ACCESS_KEY 또는 BITHUMB_SECRET_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    elif bithumb_api is None:
        logger.error("Bithumb API 객체 초기화 실패로 프로그램을 시작할 수 없습니다.")
    else:
        logger.info("실거래 모드로 실행됩니다. (Bithumb)")
        if not DISCORD_WEBHOOK_URL:
            logger.warning("DISCORD_WEBHOOK_URL이 설정되지 않아 디스코드 알림이 비활성화될 수 있습니다.")
        run_trading()
    logger.info("/main (Bithumb version)\n")