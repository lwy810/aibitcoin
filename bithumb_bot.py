import os  # 운영체제 관련 기능 사용
import time  # 시간 지연 및 시간 관련 기능
import logging  # 로깅 기능
import numpy as np  # 수치 계산
from datetime import datetime, timedelta  # 날짜/시간 처리
from dotenv import load_dotenv  # 환경변수 관리
import requests  # HTTP 요청
import sys  # 시스템 관련 기능
import jwt  # JWT 토큰
import hashlib  # 해시 함수
import uuid  # UUID 생성
from urllib.parse import urlencode  # URL 인코딩
import json  # JSON 처리


# ec2 또는 lightsail 서버에서 실행

# 1. ubuntu 22.04 서버 환경 설치
# 2. python3 설치 확인(기본 3.10 버전 설치되어 있음)
# python3 --version
# 3. pip 설치 확인
# pip3 --version
# 미설치시 설치하기
# sudo apt update
# sudo apt install python3-pip

# bot 실행하기
# pip install python-dotenv requests PyJWT numpy
# python3 bithumb_bot.py

# 크론탭 설정
# crontab -e
# crontab -l
# * * * * * cd /home/ubuntu/myapp && /usr/bin/python3 bithumb_bot.py >> cron.log 2>&1
# chmod +x /home/ubuntu/myapp/bithumb_bot.py
# tail -f /home/ubuntu/myapp/cron.log
 

# streamlit 설치 및 실행
# pip install steamlit pandas
# ~/.local/bin/streamlit run test01.py
# nohup streamlit run bot_dashboard_cron.py --server.port 8502 2>&1 &
# py -m streamlit run bot_dashboard_cron.py --server.port 8502



# Python 3.12 sqlite3 호환성을 위한 설정
import sqlite3
sqlite3.register_adapter(datetime, lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
sqlite3.register_converter("timestamp", lambda x: datetime.strptime(x.decode(), '%Y-%m-%d %H:%M:%S'))

# .env 파일에서 환경변수 로드
load_dotenv()

# API 키 설정
API_KEY = os.environ.get("BITHUMB_API_KEY")  # 빗썸 API 키
SECRET_KEY = os.environ.get("BITHUMB_SECRET_KEY")  # 빗썸 시크릿 키
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")  # 디스코드 웹훅 URL

# 빗썸 API URL
apiUrl = 'https://api.bithumb.com'


# 박스권
# 1425원  고점
# 1305원  저점

# 고저 120원 / 50구간 = 2.4원
# 일봉평균 고저 10원
# 하루 평균 거래 횟수 약 3-4회


# 상수설정 - 빗썸
TICKER = "KRW-DOGE"  # 코인 심볼
BASE_PRICE = 380  # 기준 가격 (사용자입력: 1400, 현재가로 설정: None)
PRICE_CHANGE = 7.2  # 가격 변동 기준(단위:원)
MAX_GRID_COUNT = 25  # 최대 그리드 수(단위:구간 1<=N<=100)
ORDER_AMOUNT = 10000  # 주문당 금액 (단위:원, 최소주문금액(빗썸) 500원)

FEE_RATE = 0.0004  # 거래 수수료(빗썸) (0.04%)
DISCORD_LOGGING = False  # 디스코드 로깅 비활성화
LOGGING_ENABLED = True  # 로깅 활성화
current_price = 0  # 현재 코인 가격
previous_price = None  # 이전 가격
grid_orders = []  # 그리드 주문 목록
trade_history = []  # 거래 내역

# SQLite 데이터베이스 설정
def get_or_create_db_file():
    """기존 DB 파일을 찾거나 새로 생성"""
    import glob
    
    # trading_history_bithumb_*.db 패턴으로 기존 파일 찾기
    existing_files = glob.glob('trading_history_bithumb_*.db')
    
    if existing_files:
        # 가장 최근 파일 사용 (파일명의 날짜시간 기준)
        latest_file = max(existing_files)
        print(f"기존 데이터베이스 파일 사용: {latest_file}")
        return latest_file
    else:
        # 기존 파일이 없으면 새로 생성
        current_datetime = datetime.now().strftime('%Y%m%d%H%M')
        new_file = f'trading_history_bithumb_{current_datetime}.db'
        print(f"새로운 데이터베이스 파일 생성: {new_file}")
        
        # 실제 DB 파일 생성 (빈 파일이라도 생성)
        try:
            conn = sqlite3.connect(new_file)
            conn.close()
            print(f"데이터베이스 파일 생성 완료: {new_file}")
        except Exception as e:
            print(f"데이터베이스 파일 생성 중 오류: {str(e)}")
        
        return new_file

# DB 파일 설정
DB_FILE = get_or_create_db_file()

def init_db():
    """데이터베이스 초기화 및 테이블 생성"""
    try:
        # 새로운 데이터베이스 파일 생성
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        print(f"데이터베이스 초기화: {DB_FILE}")
        
        # GRID(구간) 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS grid (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            grid_level INTEGER,
            buy_price_target REAL,
            sell_price_target REAL,
            order_krw_amount REAL,
            is_bought BOOLEAN,
            actual_bought_volume REAL,
            actual_buy_fill_price REAL,
            timestamp TEXT DEFAULT (datetime('now', 'localtime'))
        )
        ''')
        
        # 거래 내역 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            buy_sell TEXT,
            grid_level INTEGER,
            price REAL,
            amount REAL,
            volume REAL,
            fee REAL,
            profit REAL,
            profit_percentage REAL,
            timestamp TEXT DEFAULT (datetime('now', 'localtime'))
        )
        ''')
        
        # 잔고 현황 테이블
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS balance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            krw_balance REAL,
            coin_balance REAL,
            coin_avg_price REAL,
            total_assets REAL,
            current_price REAL,
            timestamp TEXT DEFAULT (datetime('now', 'localtime'))
        )
        ''')
        
        conn.commit()
        print("데이터베이스 초기화 완료")
    except Exception as e:
        print(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
    finally:
        conn.close()

# 빗썸 API 헬퍼 함수들
def GetCoinAmount(balances, ticker, type="ALL"):
    """해당 코인의 보유 수량을 얻어온다"""
    CoinAmount = 0.0
    for value in balances:
        realTicker = value['unit_currency'] + "-" + value['currency']
        
        if ticker == "KRW":
            realTicker = value['currency']

        if ticker.lower() == realTicker.lower():
            CoinAmount = float(value['balance']) 
            if type == "ALL":
                CoinAmount += float(value['locked'])
            break
    return CoinAmount

def GetBalances():
    """잔고가져오기"""
    # Generate access token
    payload = {
        'access_key': API_KEY,
        'nonce': str(uuid.uuid4()),
        'timestamp': round(time.time() * 1000)
    }
    jwt_token = jwt.encode(payload, SECRET_KEY)
    authorization_token = 'Bearer {}'.format(jwt_token)
    headers = {
        'Authorization': authorization_token
    }

    result = None
    try:
        # Call API
        response = requests.get(apiUrl + '/v1/accounts', headers=headers)
        # handle to success or fail
        # print(response.status_code)
        
        result = response.json()
        # print(response.json())
    except Exception as err:
        # handle exception
        print(err)
    
    return result

def BuyCoinMarket(ticker, money):
    """시장가 매수한다. 2초뒤 잔고 데이타 리스트를 리턴한다."""
    time.sleep(0.05)

    # Set API parameters
    requestBody = dict(market=ticker, side='bid', price=money, ord_type='price')

    # Generate access token
    query = urlencode(requestBody).encode()
    hash = hashlib.sha512()
    hash.update(query)
    query_hash = hash.hexdigest()
    payload = {
        'access_key': API_KEY,
        'nonce': str(uuid.uuid4()),
        'timestamp': round(time.time() * 1000), 
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }   
    jwt_token = jwt.encode(payload, SECRET_KEY)
    authorization_token = 'Bearer {}'.format(jwt_token)
    headers = {
        'Authorization': authorization_token,
        'Content-Type': 'application/json'
    }

    try:
        # Call API
        response = requests.post(apiUrl + '/v1/orders', data=json.dumps(requestBody), headers=headers)
        # handle to success or fail
        print(response.status_code)
        print(response.json())
    except Exception as err:
        # handle exception
        print(err)
        
    time.sleep(2.0)
    
    # 내가 가진 잔고 데이터를 다 가져온다.
    balances = GetBalances()
    return balances

def SellCoinMarket(ticker, volume):
    """시장가 매도한다. 2초뒤 잔고 데이타 리스트를 리턴한다."""
    time.sleep(0.05)

    # Set API parameters
    requestBody = dict(market=ticker, side='ask', volume=volume, ord_type='market')

    # Generate access token
    query = urlencode(requestBody).encode()
    hash = hashlib.sha512()
    hash.update(query)
    query_hash = hash.hexdigest()
    payload = {
        'access_key': API_KEY,
        'nonce': str(uuid.uuid4()),
        'timestamp': round(time.time() * 1000), 
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }   
    jwt_token = jwt.encode(payload, SECRET_KEY)
    authorization_token = 'Bearer {}'.format(jwt_token)
    headers = {
        'Authorization': authorization_token,
        'Content-Type': 'application/json'
    }

    try:
        # Call API
        response = requests.post(apiUrl + '/v1/orders', data=json.dumps(requestBody), headers=headers)
        # handle to success or fail
        print(response.status_code)
        print(response.json())
    except Exception as err:
        # handle exception
        print(err)
        
    time.sleep(2.0)
    
    # 내가 가진 잔고 데이터를 다 가져온다.
    balances = GetBalances()
    return balances

def get_current_price_bithumb():
    """빗썸에서 현재 가격 조회"""
    try:
        # 빗썸 공개 API로 현재가 조회
        url = f"https://api.bithumb.com/public/ticker/{TICKER}_KRW"
        response = requests.get(url)
        data = response.json()
        
        if data['status'] == '0000':  # 성공
            current_price = float(data['data']['closing_price'])
            return current_price
        else:
            print(f"빗썸 가격 조회 실패: {data}")
            return None
    except Exception as e:
        print(f"빗썸 가격 조회 중 오류: {str(e)}")
        return None

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

log_file = f"bithumb_bot_grid_cron.log"

# 로깅 설정 - LOGGING_ENABLED 변수에 따라 로깅 레벨 조정
if LOGGING_ENABLED:
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
    file_handler.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    logger.info("로깅 설정 완료")
else:
    # 로깅 비활성화 - CRITICAL 레벨로 설정하여 대부분의 로그 차단
    logging.basicConfig(level=logging.CRITICAL)
    logger = logging.getLogger()
    logger.disabled = True

def save_trade(trade_data):
    """거래 내역을 데이터베이스에 저장"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO trades (
            ticker, buy_sell, grid_level, price, amount, 
            volume, fee, profit, profit_percentage, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            TICKER,  # ticker 추가
            trade_data['type'],  # buy_sell 컬럼에 type 값 저장
            trade_data['grid_level'],
            trade_data['price'],
            trade_data['amount'],
            trade_data['volume'],
            trade_data.get('fee', 0),
            trade_data.get('profit', 0),
            trade_data.get('profit_percentage', 0),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 문자열로 변환
        ))
        
        conn.commit()
    except Exception as e:
        print(f"거래 내역 저장 중 오류 발생: {str(e)}")
    finally:
        conn.close()

def save_balance(balance_data):
    """잔고 현황을 데이터베이스에 저장"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # ticker 코인의 KRW 환산 가치 계산
        coin_krw_value = balance_data['coin'] * current_price if balance_data['coin'] > 0 and current_price > 0 else 0
        
        cursor.execute('''
        INSERT INTO balance_history (
            ticker, timestamp, krw_balance, coin_balance, 
            coin_avg_price, total_assets, current_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            TICKER,  # ticker 추가
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # 문자열로 변환
            coin_krw_value,  # ticker 코인의 KRW 환산 가치
            balance_data['coin'],
            balance_data['coin_avg_price'],
            coin_krw_value,  # total_assets도 ticker 코인 관련 자산만
            current_price
        ))
        
        conn.commit()
    except Exception as e:
        print(f"잔고 현황 저장 중 오류 발생: {str(e)}")
    finally:
        conn.close()

def save_grid(grid_data):
    """그리드 상태를 데이터베이스에 업데이트"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # 해당 그리드 레벨의 최신 레코드 확인 (ticker 조건 추가)
        cursor.execute('''
        SELECT id FROM grid 
        WHERE grid_level = ? AND ticker = ?
        ORDER BY timestamp DESC 
        LIMIT 1
        ''', (grid_data['level'], TICKER))
        
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
                timestamp = ?
            WHERE id = ?
            ''', (
                grid_data['buy_price_target'],
                grid_data['sell_price_target'],
                grid_data['order_krw_amount'],
                grid_data['is_bought'],
                grid_data['actual_bought_volume'],
                grid_data['actual_buy_fill_price'],
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # 문자열로 변환
                result[0]
            ))
            print(f"그리드 레벨 {grid_data['level']} 상태 업데이트 완료")
        else:
            # 기존 레코드가 없으면 새로 삽입 (ticker 포함)
            cursor.execute('''
            INSERT INTO grid (
                ticker, grid_level, buy_price_target, sell_price_target,
                order_krw_amount, is_bought, actual_bought_volume,
                actual_buy_fill_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                TICKER,  # ticker 추가
                grid_data['level'],
                grid_data['buy_price_target'],
                grid_data['sell_price_target'],
                grid_data['order_krw_amount'],
                grid_data['is_bought'],
                grid_data['actual_bought_volume'],
                grid_data['actual_buy_fill_price']
            ))
            print(f"그리드 레벨 {grid_data['level']} 새 상태 저장 완료")
        
        conn.commit()
    except Exception as e:
        print(f"그리드 상태 저장 중 오류 발생: {str(e)}")
    finally:
        conn.close()

def get_current_price():
    """현재 가격 조회"""
    logger.info("get_current_price")
    global current_price, previous_price

    try:
        ticker_price = get_current_price_bithumb()
        if ticker_price is None:
            logger.error("빗썸에서 가격 조회 실패")
            return None
            
        if 'previous_price' not in globals() or previous_price is None:
            previous_price = ticker_price
        if 'current_price' not in globals() or current_price == 0:
            current_price = ticker_price

        price_change_val = 0
        change_percentage = 0
        if previous_price is not None and previous_price > 0:
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
                elif grid_orders and ticker_price < grid_orders[-1]['buy_price_target'] - PRICE_CHANGE:
                  grid_level_info = "최하단 매수 구간 미만"

        price_msg = f"현재 {TICKER} 가격: {ticker_price:,.2f}원, 기준가: {base_price_str}, {grid_level_info}, ({sign}{change_percentage:.2f}%), {sign}{price_change_val:.2f}원 {'상승' if price_change_val >= 0 else '하락'}"
        logger.info(price_msg)
        
        previous_price = ticker_price
        current_price = ticker_price

        logger.info("/get_current_price\n")
        return ticker_price

    except Exception as e:
        logger.error(f"가격 조회 중 오류 발생: {str(e)}")
        return None

def calculate_avg_buy_price_from_db():
    """DB의 그리드 상태를 기반으로 현재 보유 코인의 평균 매수가를 계산합니다."""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # 각 그리드 레벨의 최신 상태만 조회하여, 현재 매수 상태(is_bought=True)인 것들만 가져옴
        cursor.execute('''
            SELECT g1.order_krw_amount, g1.actual_bought_volume
            FROM grid g1
            INNER JOIN (
                SELECT grid_level, MAX(timestamp) as max_timestamp
                FROM grid 
                WHERE ticker = ?
                GROUP BY grid_level
            ) g2 ON g1.grid_level = g2.grid_level AND g1.timestamp = g2.max_timestamp
            WHERE g1.ticker = ? AND g1.is_bought = 1 AND g1.actual_bought_volume > 0
        ''', (TICKER, TICKER))

        bought_grids = cursor.fetchall()
        
        if not bought_grids:
            return 0

        total_invested_krw = 0
        total_bought_volume = 0

        for amount, volume in bought_grids:
            total_invested_krw += amount
            total_bought_volume += volume

        if total_bought_volume > 0:
            avg_price = total_invested_krw / total_bought_volume
            return avg_price
        else:
            return 0

    except Exception as e:
        logger.error(f"DB에서 평균 매수가 계산 중 오류 발생: {str(e)}")
        return 0
    finally:
        if conn:
            conn.close()

def get_balance():
    """계좌 잔고 조회"""
    logger.info("get_balance")

    try:
        balances = GetBalances()
        if not balances:
            logger.error("잔고 조회 실패")
            return None
            
        krw_balance = GetCoinAmount(balances, "KRW", "ALL")
        coin_balance = GetCoinAmount(balances, f"KRW-{TICKER}", "ALL")
        
        # 빗썸에서는 평균 매수가를 직접 제공하지 않으므로, DB의 그리드 상태를 기반으로 직접 계산
        coin_avg_price = 0
        if coin_balance > 0:
            coin_avg_price = calculate_avg_buy_price_from_db()

        logger.info(f"보유 원화: {krw_balance:,.0f}원")
        logger.info(f"보유 {TICKER}: {coin_balance:.8f} {TICKER}")

        current_coin_value = 0
        if coin_balance > 0 and current_price > 0:
            current_coin_value = coin_balance * current_price
            total_investment = coin_balance * coin_avg_price
            profit = current_coin_value - total_investment
            profit_percentage = (profit / total_investment) * 100 if total_investment > 0 else 0
            
            logger.info(f"{TICKER} 평균 매수가: {coin_avg_price:,.2f}원")
            logger.info(f"평가 금액: {current_coin_value:,.0f}원")
            logger.info(f"수익금: {profit:+,.0f}원 ({profit_percentage:+.2f}%)")

        total_assets = krw_balance + current_coin_value
        logger.info(f"총 자산: {total_assets:,.0f}원")

        logger.info("/get_balance\n")
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
        # 현재가를 기준가로 설정
        BASE_PRICE = current_market_price
        logger.info(f"현재 시장 가격: {current_market_price:,.2f}원")
        logger.info(f"기준 가격 설정 (현재가 기준): {BASE_PRICE:,.2f}원")

    # 즉시 시장가 매수 제거 - 그리드 조건을 만족할 때만 매수 실행
    logger.info("그리드 조건을 만족할 때 매수를 실행합니다.")

    grid_orders = []
    
    # 현재가 위아래로 그리드 생성
    for i in range(MAX_GRID_COUNT):
        # 현재가 아래쪽 그리드들 (매수 구간)
        buy_target_price = BASE_PRICE - (i * PRICE_CHANGE)
        sell_target_price = buy_target_price + PRICE_CHANGE

        grid = {
            'level': i + 1,
            'buy_price_target': buy_target_price,
            'sell_price_target': sell_target_price,
            'buy_price_min': buy_target_price - PRICE_CHANGE,  # 매수 구간 하한
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
            f"{grid['level']}차: 매수구간 {grid['buy_price_min']:,.2f}~{grid['buy_price_target']:,.2f}원, 매도 목표가 {grid['sell_price_target']:,.2f}원")

    logger.info("/create_grid_orders\n")
    return True



def buy_coin(grid_level):
    """지정된 그리드 레벨에서 코인 시장가 매수"""
    logger.info(f"buy_coin (Level {grid_level})")
    
    global current_price

    # 그리드 레벨에 해당하는 그리드 정보 가져오기
    grid = grid_orders[grid_level - 1]

    # 이미 매수된 그리드인지 확인
    if grid['is_bought']:
        logger.info(f"레벨 {grid_level}은 이미 매수 상태입니다.")
        logger.info("/buy_coin\n")
        return False
    
    # 매수 전 데이터베이스에서 최신 상태 재확인
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT is_bought, actual_bought_volume, actual_buy_fill_price
        FROM grid 
        WHERE grid_level = ? AND ticker = ?
        ORDER BY timestamp DESC 
        LIMIT 1
        ''', (grid_level, TICKER))
        
        result = cursor.fetchone()
        
        if result:
            db_is_bought, db_bought_volume, db_buy_price = result
            if db_is_bought:
                logger.warning(f"레벨 {grid_level} 매수 중단: 데이터베이스에서 이미 매수 상태 확인됨")
                grid['is_bought'] = True
                grid['actual_bought_volume'] = db_bought_volume or 0.0
                grid['actual_buy_fill_price'] = db_buy_price or 0.0
                logger.info("/buy_coin\n")
                return False
        
        conn.close()
    except Exception as e:
        logger.error(f"매수 전 DB 상태 확인 중 오류: {str(e)}")

    try:
        # 잔고 조회
        balances = GetBalances()
        if not balances:
            logger.error("잔고 조회 실패 (buy_coin)")
            return False
            
        krw_balance = GetCoinAmount(balances, "KRW", "ALL")
        order_krw_amount = grid['order_krw_amount']
        
        # 잔고가 부족한 경우 매수 중단
        if krw_balance < order_krw_amount:
            logger.warning(f"잔액 부족: 매수 불가 (필요: {order_krw_amount:,}원, 보유: {krw_balance:,}원)")
            return False

        # 시장가 매수 시점의 현재 가격 사용 (체결 예상 가격)
        actual_fill_price = current_price 
        
        # 유효하지 않은 가격으로는 매수 불가 (0 이하 가격 방지)
        if actual_fill_price <= 0:
            logger.error(f"유효하지 않은 시장 가격 ({actual_fill_price})으로 매수 불가.")
            return False

        # 수수료 계산 (빗썸 수수료 0.04% 적용)
        # 예시: 10,000원 주문 시 수수료 4원
        fee_paid_krw = order_krw_amount * FEE_RATE
        
        # 실제 코인 구매에 사용되는 금액 = 주문 금액 - 수수료
        # 예시: 10,000원 - 4원 = 9,996원
        krw_for_coin_purchase = order_krw_amount * (1 - FEE_RATE)
        
        # 실제 매수될 코인 수량 계산 = 구매 금액 / 코인 가격
        # 예시: 9,995원 / 3,600원 = 2.776388... XRP
        actual_bought_volume = krw_for_coin_purchase / actual_fill_price

        try:
            # 빗썸 API를 통한 실제 시장가 매수 주문 실행
            logger.info(f"실제 시장가 매수 시도: {order_krw_amount:,.0f} KRW")
            order_response = BuyCoinMarket(f"KRW-{TICKER}", order_krw_amount)
            
            if not order_response:
                logger.error("매수 주문 실패")
                return False
            
            logger.info("매수 주문 성공")
            
            # 체결 후 잔고 업데이트된 정보로 실제 매수량 확인
            time.sleep(2)
            new_balances = GetBalances()
            if new_balances:
                new_coin_balance = GetCoinAmount(new_balances, f"KRW-{TICKER}", "ALL")
                # 실제 매수된 수량은 API 응답이나 잔고 변화량으로 확인해야 하지만
                # 여기서는 계산된 값을 사용
                logger.info(f"매수 체결 정보: 수량 {actual_bought_volume:.8f}, 가격 {actual_fill_price:,.2f}원")

        except Exception as e:
            logger.error(f"매수 주문 API 호출 중 오류: {str(e)}")
            return False

        # 매수 완료 후 그리드 상태 업데이트
        grid['is_bought'] = True                           # 매수 완료 상태로 변경
        grid['actual_bought_volume'] = actual_bought_volume  # 실제 매수된 수량 저장
        grid['actual_buy_fill_price'] = actual_fill_price    # 실제 매수 가격 저장
        save_grid(grid)  # 매수 후 그리드 상태를 데이터베이스에 저장

        # 거래 내역 데이터 구성
        trade = {
            'type': 'buy',                                      # 거래 유형: 매수
            'grid_level': grid_level,                          # 그리드 레벨
            'price': actual_fill_price,                        # 실제 체결 가격
            'amount': order_krw_amount,                        # 총 주문 금액 (KRW)
            'volume': actual_bought_volume,                    # 실제 매수된 코인 수량
            'fee': fee_paid_krw,                              # 지불된 수수료 (KRW)
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 거래 시간
        }
        
        # 거래 내역을 메모리와 데이터베이스에 저장
        trade_history.append(trade)
        save_trade(trade)

        # 매수 완료 로그 메시지 출력
        logger.info(f"매수 완료 (L{grid_level}): {actual_bought_volume:.8f} {TICKER} @ {actual_fill_price:,.2f}원 "
                   f"(주문액: {order_krw_amount:,.0f}원, 수수료: {fee_paid_krw:,.2f}원)")
        
        # 디스코드 알림 전송 (설정된 경우)
        if DISCORD_LOGGING:
            discord_logger.send(f"매수 완료 (L{grid_level}): {actual_bought_volume:.8f} {TICKER} @ {actual_fill_price:,.2f}원", "INFO")
        
        get_balance()
        
        logger.info("/buy_coin\n")
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
    
    # 매도 전 데이터베이스에서 최신 상태 재확인
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # 해당 그리드 레벨의 최신 상태 확인
        cursor.execute('''
        SELECT is_bought, actual_bought_volume, actual_buy_fill_price
        FROM grid 
        WHERE grid_level = ? AND ticker = ?
        ORDER BY timestamp DESC 
        LIMIT 1
        ''', (grid_level, TICKER))
        
        result = cursor.fetchone()
        
        if result:
            db_is_bought, db_bought_volume, db_buy_price = result
            if not db_is_bought or db_bought_volume <= 0:
                logger.warning(f"레벨 {grid_level} 매도 중단: 데이터베이스에서 매수 상태가 아님")
                grid['is_bought'] = False
                grid['actual_bought_volume'] = 0.0
                grid['actual_buy_fill_price'] = 0.0
                logger.info("/sell_coin\n")
                return False
            else:
                grid['is_bought'] = True
                grid['actual_bought_volume'] = db_bought_volume or 0.0
                grid['actual_buy_fill_price'] = db_buy_price or 0.0
        
        conn.close()
    except Exception as e:
        logger.error(f"매도 전 DB 상태 확인 중 오류: {str(e)}")

    # 매도할 코인 수량 = 이전에 매수했던 실제 수량
    volume_to_sell = grid['actual_bought_volume']
    
    # 매도할 수량이 0 이하인 경우 매도 불가
    if volume_to_sell <= 0:
        logger.warning(f"레벨 {grid_level} 매도할 코인 수량 없음 ({volume_to_sell:.8f}).")
        return False
        
    try:
        # 잔고 조회
        balances = GetBalances()
        if not balances:
            logger.error("잔고 조회 실패 (sell_coin)")
            return False
            
        coin_balance = GetCoinAmount(balances, f"KRW-{TICKER}", "ALL")

        if coin_balance < volume_to_sell:
            logger.warning(f"{TICKER} 보유량 부족: 매도 불가 (필요: {volume_to_sell:.8f}, 보유: {coin_balance:.8f})")
            return False

        actual_fill_price = current_price
        
        if actual_fill_price <= 0:
            logger.error(f"유효하지 않은 시장 가격 ({actual_fill_price})으로 매도 불가.")
            return False

        # 매도 금액 계산
        gross_sell_krw = volume_to_sell * actual_fill_price
        
        # 수수료 계산 (매도 금액의 0.05%)
        # 예시: 10,048원 * 0.0005 = 5.024원
        fee_paid_krw = gross_sell_krw * FEE_RATE
        
        # 실제 받을 금액 = 총 매도 금액 - 수수료
        # 예시: 10,048원 - 5.024원 = 10,042.976원
        net_sell_krw_received = gross_sell_krw * (1 - FEE_RATE)

        try:
            # 빗썸 API를 통한 실제 시장가 매도 주문 실행
            logger.info(f"실제 시장가 매도 시도: {volume_to_sell:.8f} {TICKER}")
            order_response = SellCoinMarket(f"KRW-{TICKER}", volume_to_sell)
            
            if not order_response:
                logger.error("매도 주문 실패")
                return False
            
            logger.info("매도 주문 성공")
            logger.info(f"매도 체결 정보: 수량 {volume_to_sell:.8f}, 가격 {actual_fill_price:,.2f}원")

        except Exception as e:
            logger.error(f"매도 주문 API 호출 중 오류: {str(e)}")
            return False

        # 수익 계산
        profit_for_this_trade = net_sell_krw_received - grid['order_krw_amount']
        profit_percentage = (profit_for_this_trade / grid['order_krw_amount']) * 100 if grid['order_krw_amount'] > 0 else 0
        
        # 매도 완료 후 그리드 상태 초기화 (다음 매수를 위해)
        grid['is_bought'] = False                    # 매수 상태 해제
        grid['actual_bought_volume'] = 0.0           # 보유 수량 초기화
        grid['actual_buy_fill_price'] = 0.0          # 매수 가격 초기화
        save_grid(grid)  # 매도 후 그리드 상태를 데이터베이스에 저장
        
        # 거래 내역 데이터 구성
        trade = {
            'type': 'sell',                                     # 거래 유형: 매도
            'grid_level': grid_level,                          # 그리드 레벨
            'price': actual_fill_price,                        # 실제 체결 가격
            'amount': net_sell_krw_received,                   # 실제 받은 KRW (수수료 제외)
            'volume': volume_to_sell,                          # 매도된 코인 수량
            'fee': fee_paid_krw,                              # 지불된 수수료 (KRW)
            'profit': profit_for_this_trade,                   # 이번 거래 수익
            'profit_percentage': profit_percentage,             # 이번 거래 수익률
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 거래 시간
        }
        
        trade_history.append(trade)
        save_trade(trade)

        logger.info(f"매도 완료 (L{grid_level}): {volume_to_sell:.8f} {TICKER} @ {actual_fill_price:,.2f}원 "
                   f"(실현금액: {net_sell_krw_received:,.0f}원, 수수료: {fee_paid_krw:,.2f}원)")
        logger.info(f"레벨 {grid_level} 거래 수익: {profit_for_this_trade:+,.0f}원 ({profit_percentage:+.2f}%)")
        
        if DISCORD_LOGGING:
            discord_logger.send(f"매도 완료 (L{grid_level}): {volume_to_sell:.8f} {TICKER} @ {actual_fill_price:,.2f}원\\n"
                               f"수익: {profit_for_this_trade:+,.0f}원 ({profit_percentage:+.2f}%)", "INFO")
        
        get_balance()
        
        logger.info("/sell_coin\n")
        return True

    except Exception as e:
        logger.error(f"매도 중 오류 발생 (L{grid_level}): {str(e)}")
        return False

def check_price_and_trade():
    """현재 가격을 확인하고 모든 그리드 주문에 대해 거래 실행"""
    logger.info("check_price_and_trade")
    global current_price

    if current_price is None or current_price <= 0:
        logger.error("유효하지 않은 현재 가격으로 거래 로직을 실행할 수 없습니다.")
        logger.info("/check_price_and_trade\n")
        return

    for i, grid in enumerate(grid_orders):
        level = grid['level']
        
        # 매수 조건: 현재 가격이 해당 그리드 구간 내에 있고 아직 매수되지 않은 경우
        if (not grid['is_bought'] and 
            grid['buy_price_min'] < current_price <= grid['buy_price_target']):
            logger.info(f"레벨 {level} 매수 조건 충족: 현재가({current_price:,.2f}원)가 구간({grid['buy_price_min']:,.2f}~{grid['buy_price_target']:,.2f}원) 내에 있음")
            logger.info(f"레벨 {level} 매수 시도")
            buy_coin(level)
            time.sleep(1)

        # 매도 조건: 현재 가격이 매도 목표가 이상이고 이미 매수된 경우
        elif grid['is_bought'] and current_price >= grid['sell_price_target']:
            logger.info(f"레벨 {level} 매도 조건 충족: 현재가({current_price:,.2f}원) >= 매도 목표가({grid['sell_price_target']:,.2f}원)")
            logger.info(f"레벨 {level} 매도 시도")
            sell_coin(level)
            time.sleep(1)
            
    logger.info("/check_price_and_trade\n")

def load_grid_from_db():
    """데이터베이스에서 그리드 상태 불러오기"""
    global grid_orders
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT DISTINCT g1.grid_level, g1.buy_price_target, g1.sell_price_target, 
               g1.order_krw_amount, g1.is_bought, g1.actual_bought_volume, 
               g1.actual_buy_fill_price
        FROM grid g1
        INNER JOIN (
            SELECT grid_level, MAX(timestamp) as max_timestamp
            FROM grid 
            WHERE ticker = ?
            GROUP BY grid_level
        ) g2 ON g1.grid_level = g2.grid_level AND g1.timestamp = g2.max_timestamp
        WHERE g1.ticker = ?
        ORDER BY g1.grid_level
        ''', (TICKER, TICKER))
        
        rows = cursor.fetchall()
        
        if rows:
            grid_orders = []
            for row in rows:
                grid_level, buy_price_target, sell_price_target, order_krw_amount, is_bought, actual_bought_volume, actual_buy_fill_price = row
                
                buy_price_min = buy_price_target - PRICE_CHANGE
                
                grid = {
                    'level': grid_level,
                    'buy_price_target': buy_price_target,
                    'sell_price_target': sell_price_target,
                    'buy_price_min': buy_price_min,
                    'order_krw_amount': order_krw_amount,
                    'is_bought': bool(is_bought),
                    'actual_bought_volume': actual_bought_volume or 0.0,
                    'actual_buy_fill_price': actual_buy_fill_price or 0.0
                }
                grid_orders.append(grid)
            
            logger.info(f"데이터베이스에서 {len(grid_orders)}개 그리드 상태 불러오기 완료")
            
            for grid in grid_orders:
                status = "매수완료" if grid['is_bought'] else "매수대기"
                logger.info(f"L{grid['level']}: {status} | 매수구간 {grid['buy_price_min']:,.2f}~{grid['buy_price_target']:,.2f}원 | 매도가 {grid['sell_price_target']:,.2f}원")
            
            return True
        else:
            logger.info("데이터베이스에서 그리드 데이터를 찾을 수 없습니다. 새로 생성합니다.")
            return False
            
    except Exception as e:
        logger.error(f"그리드 상태 불러오기 중 오류 발생: {str(e)}")
        return False
    finally:
        conn.close()

def run_trading():
    logger.info("run_trading - Bithumb Crontab 실행")
    global current_price, DISCORD_LOGGING

    # 데이터베이스 초기화
    init_db()
    
    logger.info(f"===== {TICKER} 자동 매매 (Bithumb Crontab 실행) =====")
    logger.info(f"티커: {TICKER}")
    logger.info(f"차수별 주문 금액: {ORDER_AMOUNT:,}원")
    logger.info(f"가격 변동 기준: {PRICE_CHANGE}원")
    logger.info(f"최대 분할 차수: {MAX_GRID_COUNT}")
    logger.info(f"거래 수수료율: {FEE_RATE * 100:.3f}%")
    logger.info(f"Discord 로깅: {'활성화' if DISCORD_LOGGING else '비활성화'}")

    try:
        logger.info("빗썸 실거래 모드 활성화됨")
        
        # 현재가 조회
        current_price = get_current_price()
        if current_price is None:
            logger.error("현재 가격 조회 실패. 빗썸 API키, 허용IP 목록을 확인하세요. 종료합니다.")
            return

        # 기존 그리드 상태 불러오기 시도
        grid_loaded = load_grid_from_db()
        
        if not grid_loaded:
            logger.info("기존 그리드가 없어 새로 생성합니다.")
            
            if BASE_PRICE is not None:
                input_base_price_for_grid = BASE_PRICE
                logger.info(f"현재 시장 가격: {current_price:,.2f}원")
                logger.info(f"그리드 기준 가격 설정 (사용자 지정): {input_base_price_for_grid:,.2f}원")
            else:
                input_base_price_for_grid = current_price
                logger.info(f"현재 시장 가격: {current_price:,.2f}원")
                logger.info(f"그리드 기준 가격 설정 (현재가 기준): {input_base_price_for_grid:,.2f}원")

            if not create_grid_orders(input_base_price_for_grid):
                logger.error("그리드 주문 생성 실패. 프로그램을 종료합니다.")
                return
        else:
            logger.info("기존 그리드 상태를 성공적으로 불러왔습니다.")

        # 한 번의 거래 체크 실행
        logger.info("===== 거래 체크 시작 =====")
        check_price_and_trade()
        
        # 잔고 확인
        get_balance()
        
        logger.info("===== 거래 체크 완료 =====")

    except KeyboardInterrupt:
        logger.info("\n사용자에 의해 거래가 중단되었습니다.")
    except Exception as e:
        logger.error(f"\n거래 중 예외 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("===== 빗썸 거래 로직 종료 =====")

    logger.info("/run_trading\n")

if __name__ == "__main__":
    logger.info("Bithumb Grid Trading Bot main")
    if not API_KEY or not SECRET_KEY:
        logger.error("빗썸 API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
    else:
        logger.info("빗썸 실거래 모드로 실행됩니다.")
        if not DISCORD_WEBHOOK_URL:
            logger.warning("DISCORD_WEBHOOK_URL이 설정되지 않아 디스코드 알림이 비활성화될 수 있습니다.")
        run_trading()
    logger.info("/Bithumb Grid Trading Bot main\n")
