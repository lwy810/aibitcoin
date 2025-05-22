# 기본 라이브러리 임포트
import os  # 운영체제 관련 기능 사용
import time  # 시간 지연 및 시간 관련 기능
import logging  # 로깅 기능
import numpy as np  # 수치 계산
from datetime import datetime, timedelta  # 날짜/시간 처리
from dotenv import load_dotenv  # 환경변수 관리
import pyupbit  # 업비트 API 연동
import requests  # HTTP 요청
import sys  # 시스템 관련 기능
import sqlite3

# Windows 환경에서만 winsound 모듈 import (소리 알림용)
if sys.platform == 'win32':
    import winsound

# .env 파일에서 환경변수 로드
load_dotenv()

# API 키 설정
ACCESS_KEY = os.environ.get("UPBIT_ACCESS_KEY")  # 업비트 액세스 키
SECRET_KEY = os.environ.get("UPBIT_SECRET_KEY")  # 업비트 시크릿 키
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")  # 디스코드 웹훅 URL

# 실거래 설정
TICKER = "KRW-XRP"  # 거래할 코인 (단위:티커)
BASE_PRICE = None  # 기준 가격 (자동 설정됨)
PRICE_CHANGE = 4  # 가격 변동 기준(단위:원)
OFFSET_GRID = 4  # 기준가로부터의 구간 오프셋(단위:구간 0<= N <=MAX_GRID_COUNT)
ORDER_AMOUNT = 5000  # 주문당 금액 (단위:원, 최소주문금액 5000원)
MAX_GRID_COUNT = 10  # 최대 그리드 수(단위:구간 1<= N <=100)
CHECK_INTERVAL = 10  # 가격 체크 간격 (단위:초)
    
FEE_RATE = 0.0005  # 거래 수수료 (0.05%)
DISCORD_LOGGING = False  # 디스코드 로깅 비활성화

# 전역 변수
current_price = 0  # 현재 코인 가격
previous_price = None  # 이전 가격
grid_orders = []  # 그리드 주문 목록
trade_history = []  # 거래 내역

# SQLite 데이터베이스 설정
DB_FILE = 'trading_history.db'

def init_db():
    """데이터베이스 초기화 및 테이블 생성"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # GRID(구간) 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS grid (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            grid_level INTEGER,
            buy_price_target REAL,
            sell_price_target REAL,
            order_krw_amount REAL,
            is_bought BOOLEAN,
            actual_bought_volume REAL,
            actual_buy_fill_price REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 거래 내역 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            grid_level INTEGER,
            price REAL,
            amount REAL,
            volume REAL,
            fee REAL,
            profit REAL,
            profit_percentage REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 잔고 현황 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS balance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            krw_balance REAL,
            coin_balance REAL,
            coin_avg_price REAL,
            total_assets REAL,
            current_price REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        logger.info("데이터베이스 초기화 완료")
    except Exception as e:
        logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
    finally:
        conn.close()

def save_trade(trade_data):
    """거래 내역을 데이터베이스에 저장"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO trades (
            timestamp, type, grid_level, price, amount, 
            volume, fee, profit, profit_percentage
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            trade_data['type'],
            trade_data['grid_level'],
            trade_data['price'],
            trade_data['amount'],
            trade_data['volume'],
            trade_data.get('fee', 0),
            trade_data.get('profit', 0),
            trade_data.get('profit_percentage', 0)
        ))
        
        conn.commit()
    except Exception as e:
        logger.error(f"거래 내역 저장 중 오류 발생: {str(e)}")
    finally:
        conn.close()

def save_balance(balance_data):
    """잔고 현황을 데이터베이스에 저장"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO balance_history (
            timestamp, krw_balance, coin_balance, 
            coin_avg_price, total_assets, current_price
        ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            balance_data['krw'],
            balance_data['coin'],
            balance_data['coin_avg_price'],
            balance_data['total_assets'],
            current_price
        ))
        
        conn.commit()
    except Exception as e:
        logger.error(f"잔고 현황 저장 중 오류 발생: {str(e)}")
    finally:
        conn.close()

def save_grid(grid_data):
    """그리드 상태를 데이터베이스에 업데이트"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # 해당 그리드 레벨의 최신 레코드 확인
        cursor.execute('''
        SELECT id FROM grid 
        WHERE grid_level = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
        ''', (grid_data['level'],))
        
        result = cursor.fetchone()
        
        if result:
            # 기존 레코드가 있으면 업데이트
            cursor.execute('''
            UPDATE grid SET
                buy_price_target = ?,
                sell_price_target = ?,
                order_krw_amount = ?,
                is_bought = ?,
                actual_bought_volume = ?,
                actual_buy_fill_price = ?,
                timestamp = CURRENT_TIMESTAMP
            WHERE id = ?
            ''', (
                grid_data['buy_price_target'],
                grid_data['sell_price_target'],
                grid_data['order_krw_amount'],
                grid_data['is_bought'],
                grid_data['actual_bought_volume'],
                grid_data['actual_buy_fill_price'],
                result[0]
            ))
            logger.info(f"그리드 레벨 {grid_data['level']} 상태 업데이트 완료")
        else:
            # 기존 레코드가 없으면 새로 삽입
            cursor.execute('''
            INSERT INTO grid (
                grid_level, buy_price_target, sell_price_target,
                order_krw_amount, is_bought, actual_bought_volume,
                actual_buy_fill_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                grid_data['level'],
                grid_data['buy_price_target'],
                grid_data['sell_price_target'],
                grid_data['order_krw_amount'],
                grid_data['is_bought'],
                grid_data['actual_bought_volume'],
                grid_data['actual_buy_fill_price']
            ))
            logger.info(f"그리드 레벨 {grid_data['level']} 새 상태 저장 완료")
        
        conn.commit()
    except Exception as e:
        logger.error(f"그리드 상태 저장 중 오류 발생: {str(e)}")
    finally:
        conn.close()

# Upbit 객체 생성
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)


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

log_file = f"{TICKER.replace('KRW-', '').lower()}_grid_trade.log"
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

def get_current_price():
    """현재 가격 조회"""
    logger.info("get_current_price")
    global current_price, previous_price

    try:
        ticker_price = pyupbit.get_current_price(TICKER)
        if 'previous_price' not in globals() or previous_price is None:
            previous_price = ticker_price
        if 'current_price' not in globals() or current_price == 0:
            current_price = ticker_price

        price_change_val = 0
        change_percentage = 0
        if previous_price is not None and previous_price > 0 : # Modified to check previous_price > 0
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
                elif grid_orders and ticker_price < grid_orders[-1]['buy_price_target'] - PRICE_CHANGE : # Check against last grid
                  grid_level_info = "최하단 매수 구간 미만"


        price_msg = f"현재 {TICKER} 가격: {ticker_price:,.2f}원, 기준가: {base_price_str}, {grid_level_info}, ({sign}{change_percentage:.2f}%), {sign}{price_change_val:.2f}원 {'상승' if price_change_val >= 0 else '하락'}"
        logger.info(price_msg)
        
        previous_price = ticker_price
        current_price = ticker_price

        logger.info("/get_current_price\\n")
        return ticker_price

    except Exception as e:
        logger.error(f"가격 조회 중 오류 발생: {str(e)}")
        return None


def get_balance():
    """계좌 잔고 조회"""
    logger.info("get_balance")

    try:
        krw_balance = upbit.get_balance("KRW")
        coin_balance = upbit.get_balance(TICKER)
        coin_avg_price = upbit.get_avg_buy_price(TICKER) if upbit.get_balance(TICKER) > 0 else 0

        logger.info(f"보유 원화: {krw_balance:,.0f}원")
        logger.info(f"보유 {TICKER}: {coin_balance:.8f} {TICKER}")

        current_coin_value = 0
        if coin_balance > 0 and current_price > 0 : # Ensure current_price is available
            current_coin_value = coin_balance * current_price
            total_investment = coin_balance * coin_avg_price
            profit = current_coin_value - total_investment
            profit_percentage = (profit / total_investment) * 100 if total_investment > 0 else 0
            
            logger.info(f"{TICKER} 평균 매수가: {coin_avg_price:,.2f}원")
            logger.info(f"평가 금액: {current_coin_value:,.0f}원")
            logger.info(f"수익금: {profit:+,.0f}원 ({profit_percentage:+.2f}%)")

        total_assets = krw_balance + current_coin_value
        logger.info(f"총 자산: {total_assets:,.0f}원")

        logger.info("/get_balance\\n")
        balance_data = {
            "krw": krw_balance,
            "coin": coin_balance,
            "coin_avg_price": coin_avg_price,
            "total_assets": total_assets
        }
        
        save_balance(balance_data)  # 잔고 현황 저장
        
        return balance_data
    except Exception as e:
        logger.error(f"잔고 조회 중 오류 발생: {str(e)}")
        return None


def create_grid_orders(input_base_price=None):
    """분할 매수/매도 그리드 주문 생성"""
    logger.info("create_grid_orders")
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
        save_grid(grid)  # 각 그리드 생성 시 저장

    logger.info(f"총 {len(grid_orders)}개의 그리드 주문 설정 생성됨")
    logger.info(f"매수/매도 가격 간격: {PRICE_CHANGE}원")
    logger.info(f"주문당 KRW 금액: {ORDER_AMOUNT:,}원")
    for grid in grid_orders:
        logger.info(
            f"{grid['level']}차: 매수 목표가 {grid['buy_price_target']:,.2f}원, 매도 목표가 {grid['sell_price_target']:,.2f}원")

    logger.info("/create_grid_orders\\n")
    return True


def play_sound(sound_type):
    """거래 알림음 재생"""
    try:
        if sys.platform != 'win32':
            logger.info(f"사운드 재생: {sound_type} (Windows 환경에서만 지원)")
            return
        if sound_type == 'buy':
            winsound.PlaySound('res/buy.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif sound_type == 'sell':
            winsound.PlaySound('res/sell.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        logger.error(f"알림음 재생 중 오류 발생: {str(e)}")
        logger.info("알림음 파일(res/buy.wav, res/sell.wav)이 res 폴더에 있는지 확인하세요.")


def buy_coin(grid_level):
    """지정된 그리드 레벨에서 코인 시장가 매수"""
    logger.info(f"buy_coin (Level {grid_level})")
    global current_price

    grid = grid_orders[grid_level - 1]

    if grid['is_bought']:
        logger.info(f"레벨 {grid_level}은 이미 매수 상태입니다.")
        logger.info("/buy_coin\\n")
        return False

    try:
        krw_balance = upbit.get_balance("KRW")
        if krw_balance is None: # Explicitly check for None
            logger.error("원화 잔고 조회 실패 (buy_coin)")
            return False
        
        order_krw_amount = grid['order_krw_amount']
        if krw_balance < order_krw_amount:
            logger.warning(f"잔액 부족: 매수 불가 (필요: {order_krw_amount:,}원, 보유: {krw_balance:,}원)")
            return False

        # 시장가 매수 시점의 가격 사용
        actual_fill_price = current_price 
        if actual_fill_price <= 0: # Prevent division by zero or invalid price
            logger.error(f"유효하지 않은 시장 가격 ({actual_fill_price})으로 매수 불가.")
            return False

        # 수수료 계산 (KRW 기준)
        fee_paid_krw = order_krw_amount * FEE_RATE
        krw_for_coin_purchase = order_krw_amount * (1 - FEE_RATE)
        actual_bought_volume = krw_for_coin_purchase / actual_fill_price

        try:
            logger.info(f"실제 시장가 매수 시도: {order_krw_amount:,.0f} KRW")
            order_response = upbit.buy_market_order(TICKER, order_krw_amount)
            if not order_response or 'uuid' not in order_response: # Check for valid response
                logger.error(f"매수 주문 실패. 응답: {order_response}")
                return False
            
            logger.info(f"매수 주문 성공: UUID {order_response.get('uuid')}")
            # 실제 체결 정보를 API 응답에서 추출
            # 주문 UUID로 체결 정보 확인
            time.sleep(1)  # 체결 정보가 업데이트될 시간을 주기 위해 약간의 지연
            try:
                order_detail = upbit.get_order(order_response.get('uuid'))
                if order_detail and 'state' in order_detail and order_detail['state'] == 'done':
                    # 체결된 주문의 경우 실제 체결 정보 사용
                    executed_volume = float(order_detail.get('executed_volume', '0'))
                    if executed_volume > 0:
                        actual_bought_volume = executed_volume
                    
                    # 평균 체결가가 있으면 사용
                    avg_price = order_detail.get('avg_price')
                    if avg_price:
                        actual_fill_price = float(avg_price)
                    
                    # 지불 수수료 정보가 있으면 사용
                    paid_fee = order_detail.get('paid_fee')
                    if paid_fee:
                        fee_paid_krw = float(paid_fee)
                
                logger.info(f"매수 체결 정보: 수량 {actual_bought_volume:.8f}, 평균가 {actual_fill_price:,.2f}원, 수수료 {fee_paid_krw:,.2f}원")
            except Exception as e:
                logger.warning(f"체결 정보 조회 중 오류 (기본 추정값 사용): {str(e)}")
                # 오류 발생 시 기존 계산 값 사용 (기본 값)
                logger.info(f"기본 추정 매수 정보: 수량 {actual_bought_volume:.8f}, 가격 {actual_fill_price:,.2f}원")

        except Exception as e:
            logger.error(f"매수 주문 API 호출 중 오류: {str(e)}")
            return False

        grid['is_bought'] = True
        grid['actual_bought_volume'] = actual_bought_volume
        grid['actual_buy_fill_price'] = actual_fill_price
        save_grid(grid)  # 매수 후 그리드 상태 저장

        trade = {
            'type': 'buy',
            'grid_level': grid_level,
            'price': actual_fill_price,
            'amount': order_krw_amount, # 총 주문 금액 (KRW)
            'volume': actual_bought_volume, # 실제 매수된 코인 양
            'fee': fee_paid_krw, # 지불된 수수료 (KRW)
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        trade_history.append(trade)
        save_trade(trade)  # 거래 내역 저장

        logger.info(f"매수 완료 (L{grid_level}): {actual_bought_volume:.8f} {TICKER} @ {actual_fill_price:,.2f}원 (주문액: {order_krw_amount:,.0f}원, 수수료: {fee_paid_krw:,.2f}원)")
        if DISCORD_LOGGING:
            discord_logger.send(f"매수 완료 (L{grid_level}): {actual_bought_volume:.8f} {TICKER} @ {actual_fill_price:,.2f}원 (주문액: {order_krw_amount:,.0f}원, 수수료: {fee_paid_krw:,.2f}원)", "INFO")
        play_sound('buy')
        get_balance()
        logger.info("/buy_coin\\n")
        return True

    except Exception as e:
        logger.error(f"매수 중 오류 발생 (L{grid_level}): {str(e)}")
        return False


def sell_coin(grid_level):
    """지정된 그리드 레벨에서 코인 시장가 매도"""
    logger.info(f"sell_coin (Level {grid_level})")
    global current_price

    grid = grid_orders[grid_level - 1]

    if not grid['is_bought']:
        logger.info(f"레벨 {grid_level}은 매수 상태가 아니므로 매도 불가.")
        return False

    volume_to_sell = grid['actual_bought_volume']
    if volume_to_sell <= 0:
        logger.warning(f"레벨 {grid_level} 매도할 코인 수량 없음 ({volume_to_sell:.8f}).")
        return False
        
    try:
        coin_balance = upbit.get_balance(TICKER)
        if coin_balance is None: # Explicitly check for None
            logger.error("코인 잔고 조회 실패 (sell_coin)")
            return False


        if coin_balance < volume_to_sell:
            logger.warning(f"{TICKER} 보유량 부족: 매도 불가 (필요: {volume_to_sell:.8f}, 보유: {coin_balance:.8f})")
            # 부분 매도 로직 추가 가능하나, 현재는 전체 매도 실패로 처리
            return False

        actual_fill_price = current_price # 시장가 매도 시점의 가격 사용
        if actual_fill_price <= 0:
            logger.error(f"유효하지 않은 시장 가격 ({actual_fill_price})으로 매도 불가.")
            return False

        gross_sell_krw = volume_to_sell * actual_fill_price
        fee_paid_krw = gross_sell_krw * FEE_RATE
        net_sell_krw_received = gross_sell_krw * (1 - FEE_RATE)

        try:
            logger.info(f"실제 시장가 매도 시도: {volume_to_sell:.8f} {TICKER}")
            order_response = upbit.sell_market_order(TICKER, volume_to_sell)
            if not order_response or 'uuid' not in order_response:
                logger.error(f"매도 주문 실패. 응답: {order_response}")
                return False
            logger.info(f"매도 주문 성공: UUID {order_response.get('uuid')}")
            
            # 실제 체결 정보를 API 응답에서 추출
            time.sleep(1)  # 체결 정보가 업데이트될 시간을 주기 위해 약간의 지연
            try:
                order_detail = upbit.get_order(order_response.get('uuid'))
                if order_detail and 'state' in order_detail and order_detail['state'] == 'done':
                    # 체결된 경우 실제 정보 사용
                    executed_volume = float(order_detail.get('executed_volume', '0'))
                    if executed_volume > 0:
                        volume_to_sell = executed_volume  # 실제 매도된 수량으로 업데이트
                    
                    # 평균 체결가가 있으면 사용
                    avg_price = order_detail.get('avg_price')
                    if avg_price:
                        actual_fill_price = float(avg_price)
                        gross_sell_krw = volume_to_sell * actual_fill_price
                        fee_paid_krw = gross_sell_krw * FEE_RATE
                        net_sell_krw_received = gross_sell_krw * (1 - FEE_RATE)
                    
                    # 지불 수수료 정보가 있으면 사용
                    paid_fee = order_detail.get('paid_fee')
                    if paid_fee:
                        fee_paid_krw = float(paid_fee)
                        # 수수료가 있으면 순 매도 금액 재계산
                        net_sell_krw_received = gross_sell_krw - fee_paid_krw
                
                logger.info(f"매도 체결 정보: 수량 {volume_to_sell:.8f}, 평균가 {actual_fill_price:,.2f}원, 수수료 {fee_paid_krw:,.2f}원")
            except Exception as e:
                logger.warning(f"매도 체결 정보 조회 중 오류 (기본 추정값 사용): {str(e)}")
                # 오류 발생 시 기존 계산 값 사용
                logger.info(f"기본 추정 매도 정보: 수량 {volume_to_sell:.8f}, 가격 {actual_fill_price:,.2f}원")

        except Exception as e:
            logger.error(f"매도 주문 API 호출 중 오류: {str(e)}")
            return False

        profit_for_this_trade = net_sell_krw_received - grid['order_krw_amount']
        profit_percentage = (profit_for_this_trade / grid['order_krw_amount']) * 100 if grid['order_krw_amount'] > 0 else 0
        
        # 그리드 상태 초기화
        grid['is_bought'] = False
        grid['actual_bought_volume'] = 0.0
        grid['actual_buy_fill_price'] = 0.0
        save_grid(grid)  # 매도 후 그리드 상태 저장
        
        trade = {
            'type': 'sell',
            'grid_level': grid_level,
            'price': actual_fill_price,
            'amount': net_sell_krw_received, # 실제 받은 KRW (수수료 제외)
            'volume': volume_to_sell,
            'fee': fee_paid_krw, # 지불된 수수료 (KRW)
            'profit': profit_for_this_trade,
            'profit_percentage': profit_percentage,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        trade_history.append(trade)
        save_trade(trade)  # 거래 내역 저장

        logger.info(f"매도 완료 (L{grid_level}): {volume_to_sell:.8f} {TICKER} @ {actual_fill_price:,.2f}원 (실현금액: {net_sell_krw_received:,.0f}원, 수수료: {fee_paid_krw:,.2f}원)")
        logger.info(f"레벨 {grid_level} 거래 수익: {profit_for_this_trade:+,.0f}원 ({profit_percentage:+.2f}%)")
        if DISCORD_LOGGING:
            discord_logger.send(f"매도 완료 (L{grid_level}): {volume_to_sell:.8f} {TICKER} @ {actual_fill_price:,.2f}원 (실현금액: {net_sell_krw_received:,.0f}원)\\n수익: {profit_for_this_trade:+,.0f}원 ({profit_percentage:+.2f}%)", "INFO")
        play_sound('sell')
        get_balance()
        logger.info("/sell_coin\n")
        return True

    except Exception as e:
        logger.error(f"매도 중 오류 발생 (L{grid_level}): {str(e)}")
        return False


def check_price_and_trade():
    """현재 가격을 확인하고 모든 그리드 주문에 대해 거래 실행"""
    logger.info("check_price_and_trade")
    global current_price # Ensure it uses the global current_price updated by get_current_price()

    # 현재 가격 업데이트 (check_price_and_trade 호출 전에 외부에서 get_current_price 호출)
    # current_price = get_current_price() # 이미 run_trading 루프에서 호출됨
    if current_price is None or current_price <= 0: # current_price 유효성 검사
        logger.error("유효하지 않은 현재 가격으로 거래 로직을 실행할 수 없습니다.")
        logger.info("/check_price_and_trade\n")
        return

    for i, grid in enumerate(grid_orders): # Iterate with index
        level = grid['level'] # level for logging
        
        # 매수 조건: 현재 가격이 매수 목표가 이하이고 아직 매수되지 않은 경우
        if not grid['is_bought'] and current_price <= grid['buy_price_target']:
            logger.info(f"레벨 {level} 매수 조건 충족: 현재가({current_price:,.2f}원) <= 매수 목표가({grid['buy_price_target']:,.2f}원)")
            buy_coin(level) # buy_coin expects level number (1-indexed)
            time.sleep(1) # 주문 처리 간격

        # 매도 조건: 현재 가격이 매도 목표가 이상이고 이미 매수된 경우
        elif grid['is_bought'] and current_price >= grid['sell_price_target']:
            logger.info(f"레벨 {level} 매도 조건 충족: 현재가({current_price:,.2f}원) >= 매도 목표가({grid['sell_price_target']:,.2f}원)")
            sell_coin(level) # sell_coin expects level number
            time.sleep(1) # 주문 처리 간격
            
    logger.info("/check_price_and_trade\n")


def run_trading():
    logger.info("run_trading")
    global current_price, DISCORD_LOGGING

    # 데이터베이스 초기화
    init_db()
    
    logger.info(f"===== {TICKER} 자동 매매 시작 =====")
    logger.info(f"티커: {TICKER}")
    logger.info(f"차수별 주문 금액: {ORDER_AMOUNT:,}원")
    logger.info(f"가격 변동 기준: {PRICE_CHANGE}원")
    logger.info(f"최대 분할 차수: {MAX_GRID_COUNT}")
    logger.info(f"가격 확인 간격: {CHECK_INTERVAL}초")
    logger.info(f"거래 수수료율: {FEE_RATE * 100:.3f}%")
    logger.info(f"Discord 로깅: {'활성화' if DISCORD_LOGGING else '비활성화'}")

    try:
        logger.info("실거래 모드 활성화됨")
        # 현재가 조회 및 초기 자산 기록 (실거래 모드)
        temp_current_price = get_current_price() # 초기 가격 조회
        if temp_current_price is None:
            logger.error("프로그램 시작 시 현재 가격 조회 실패. 종료합니다.")
            return

        input_base_price_for_grid = temp_current_price + (PRICE_CHANGE * OFFSET_GRID)
        logger.info(f"현재 시장 가격: {temp_current_price:,.2f}원")
        logger.info(f"그리드 기준 가격 자동 설정 (현재가 + {OFFSET_GRID}구간): {input_base_price_for_grid:,.2f}원")

        if not create_grid_orders(input_base_price_for_grid):
            logger.error("그리드 주문 생성 실패. 프로그램을 종료합니다.")
            return

        logger.info(f"===== 매매 루프 시작 =====")
        cycle_count = 0
        while True:
            cycle_count = (cycle_count % 10000) + 1
            logger.info(f"===== 사이클 #{cycle_count} =====")
            
            current_price = get_current_price() # 루프 시작 시 가격 업데이트
            if current_price is None:
                logger.warning("현재 가격 조회 실패. 다음 사이클까지 대기합니다.")
                time.sleep(CHECK_INTERVAL)
                continue

            check_price_and_trade() # 업데이트된 current_price 사용
            
            logger.info(f"{CHECK_INTERVAL}초 대기...")
            get_balance()  # 이 부분이 주기적으로 실행되어야 함
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        logger.info("\n사용자에 의해 거래가 중단되었습니다.")
    except Exception as e:
        logger.error(f"\n거래 중 예외 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("===== 거래 로직 종료 =====")
        # 최종 잔고 상태 한번 더 확인 및 로깅
        if current_price is None or current_price == 0: # 최종 결과 표시 위한 가격 확인
            current_price = get_current_price() or 0 # 마지막 시도 또는 0
        
        get_balance() # 최종 잔고 출력

        buy_count = sum(1 for trade in trade_history if trade['type'] == 'buy')
        sell_count = sum(1 for trade in trade_history if trade['type'] == 'sell')
        logger.info(f"\n최종 거래 요약: 총 {buy_count + sell_count}회 체결 (매수: {buy_count}회, 매도: {sell_count}회)")

    logger.info("/run_trading\n")


if __name__ == "__main__":
    logger.info("main")
    if not ACCESS_KEY or not SECRET_KEY: # 실거래 시에만 키 확인
        logger.error("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
    else:
        logger.info("실거래 모드로 실행됩니다.")
        if not DISCORD_WEBHOOK_URL:
            logger.warning("DISCORD_WEBHOOK_URL이 설정되지 않아 디스코드 알림이 비활성화될 수 있습니다.")
        run_trading()
    logger.info("/main\n")