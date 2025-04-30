import os
import time
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import numpy as np
import threading

# .env 파일에서 환경변수 로드
load_dotenv()

# 업비트 API 키 설정
access_key = os.environ.get("UPBIT_ACCESS_KEY")
secret_key = os.environ.get("UPBIT_SECRET_KEY")

# 테스트 모드 설정
TEST_MODE = True  # True: 실제 주문 실행 안 함, False: 실제 주문 실행

# 업비트 API 객체 생성 부분 개선
try:
    upbit = pyupbit.Upbit(access_key, secret_key)
    print("업비트 API 객체가 생성되었습니다.")
except Exception as e:
    print(f"업비트 API 객체 생성 오류: {e}")
    exit(1)

# 세븐스플릿 전략 파라미터
TICKER = "KRW-BTC"  # 비트코인 티커
SPLIT_COUNT = 7  # 분할 횟수
MIN_ORDER_PRICE = 5000  # 최소 주문 금액 (원)
PRICE_DROP_PERCENT = 5.0  # 추가 매수 시점 (이전 매수 대비 하락률)
PROFIT_PERCENT = 3.0  # 익절 시점 (매수 대비 상승률)
MONITOR_INTERVAL = 60  # 모니터링 간격 (초)

# 분할 매수 및 익절 기록 저장
# (순서, 매수가격, 매수시간, 매수수량, 매수금액, 익절가격, 익절여부)
buy_records = []
monitoring_active = False
monitoring_thread = None


def get_current_price(ticker):
    """현재 시장 가격을 조회하는 함수"""
    try:
        price = pyupbit.get_current_price(ticker)
        return price
    except Exception as e:
        print(f"가격 조회 오류: {e}")
        return None


def get_account_balance():
    """계정 잔액 정보를 조회하는 함수"""
    try:
        # 원화 잔액
        krw_balance = upbit.get_balance("KRW")
        if krw_balance is None:
            krw_balance = 0
            print("원화 잔액을 조회할 수 없습니다.")

        # 비트코인 잔액
        btc_balance = upbit.get_balance(TICKER)
        if btc_balance is None:
            btc_balance = 0
            print("비트코인 잔액을 조회할 수 없습니다.")

        # 비트코인의 현재 가격
        current_price = get_current_price(TICKER)

        # 비트코인 평가금액 (KRW)
        btc_value = 0
        if btc_balance > 0 and current_price is not None:
            btc_value = btc_balance * current_price

        return {
            "krw_balance": krw_balance,
            "btc_balance": btc_balance,
            "btc_value": btc_value,
            "total_value": krw_balance + btc_value,
            "current_price": current_price
        }
    except Exception as e:
        print(f"계정 잔액 조회 오류: {e}")
        return {
            "krw_balance": 0,
            "btc_balance": 0,
            "btc_value": 0,
            "total_value": 0,
            "current_price": 0
        }


def detect_first_buy():
    """업비트 계정에서 현재 비트코인 잔고를 확인하여 첫 매수로 설정하는 함수"""
    if len(buy_records) > 0:
        print("이미 매수 기록이 있습니다. 초기화 후 다시 시도하세요.")
        return False

    try:
        # 현재 보유 비트코인 확인
        btc_balance = upbit.get_balance(TICKER)
        if btc_balance is None or btc_balance <= 0:
            print("비트코인 잔고가 없습니다. 업비트 앱에서 첫 매수를 먼저 진행해주세요.")
            return False

        # 현재 가격 조회
        current_price = get_current_price(TICKER)
        if current_price is None:
            print("현재 가격을 조회할 수 없습니다.")
            return False

        # 매수 금액 추정 (현재 가격 * 보유량)
        estimated_amount = current_price * btc_balance

        print(f"\n===== 첫 매수 감지 =====")
        print(f"보유 비트코인: {btc_balance:.8f} BTC")
        print(f"현재 가격: {current_price:,.0f}원")
        print(f"예상 매수 금액: {estimated_amount:,.0f}원")

        # 첫 매수 기록 저장
        buy_records.append({
            "order": 1,
            "price": current_price,
            "time": datetime.now(),
            "volume": btc_balance,
            "amount": estimated_amount,
            "profit_price": current_price * (1 + PROFIT_PERCENT / 100),
            "profit_taken": False
        })

        print("\n첫 매수가 감지되어 등록되었습니다. 이후 매수는 자동으로 진행됩니다.")
        print(f"다음 매수 가격: {current_price * (1 - PRICE_DROP_PERCENT / 100):,.0f}원 (현재 가격에서 {PRICE_DROP_PERCENT}% 하락 시)")
        print(f"익절 가격: {current_price * (1 + PROFIT_PERCENT / 100):,.0f}원 (매수 가격에서 {PROFIT_PERCENT}% 상승 시)")

        return True

    except Exception as e:
        print(f"첫 매수 감지 중 오류 발생: {e}")
        return False


def execute_buy(order_num, buy_amount):
    """매수 주문을 실행하는 함수"""
    try:
        # 현재 가격 조회
        current_price = get_current_price(TICKER)
        if current_price is None:
            print("현재 가격을 조회할 수 없어 주문을 건너뜁니다.")
            return None

        print(f"\n[{order_num}/{SPLIT_COUNT}] 매수 주문 진행 중...")
        print(f"현재 가격: {current_price:,.0f}원")
        print(f"매수 금액: {buy_amount:,.0f}원")

        # 구매 수량 계산
        buy_volume = buy_amount / current_price

        if TEST_MODE:
            # 테스트 모드 - 실제 주문 실행 안 함
            print("[테스트 모드] 실제 매수 주문은 실행되지 않았습니다.")
            order = {
                'uuid': f'dummy-uuid-{uuid.uuid4()}',
                'side': 'bid',
                'ord_type': 'price',
                'price': buy_amount,
                'state': 'done',
                'market': TICKER,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'volume': buy_volume,
                'remaining_volume': 0,
                'reserved_fee': 0,
                'remaining_fee': 0,
                'paid_fee': 0,
                'locked': buy_amount,
                'executed_volume': buy_volume,
                'trades_count': 1
            }
        else:
            # 실제 시장가 매수 주문
            order = upbit.buy_market_order(TICKER, buy_amount)

            # 주문 상태 확인을 위한 대기
            time.sleep(1)
            order_info = upbit.get_order(order['uuid'])
            if order_info and 'state' in order_info:
                print(f"주문 상태: {order_info['state']}")
                order = order_info

        # 매수 기록 저장
        buy_records.append({
            "order": order_num,
            "price": current_price,
            "time": datetime.now(),
            "volume": buy_volume,
            "amount": buy_amount,
            "profit_price": current_price * (1 + PROFIT_PERCENT / 100),
            "profit_taken": False
        })

        print(f"매수 완료 - {order_num}차 매수")
        print(f"매수 가격: {current_price:,.0f}원")
        print(f"매수 수량: {buy_volume:.8f} BTC")
        print(f"익절 가격: {current_price * (1 + PROFIT_PERCENT / 100):,.0f}원")

        if order_num < SPLIT_COUNT:
            print(
                f"다음 매수 가격: {current_price * (1 - PRICE_DROP_PERCENT / 100):,.0f}원 (현재 가격에서 {PRICE_DROP_PERCENT}% 하락 시)")

        return order

    except Exception as e:
        print(f"매수 주문 중 오류 발생: {e}")
        return None


def execute_sell(buy_record):
    """매도 주문을 실행하는 함수"""
    try:
        # 익절할 수량
        sell_volume = buy_record["volume"]
        current_price = get_current_price(TICKER)

        print(f"\n===== {buy_record['order']}차 매수분 익절 =====")
        print(f"매수 가격: {buy_record['price']:,.0f}원")
        print(f"익절 가격: {current_price:,.0f}원")
        print(f"수익률: {((current_price / buy_record['price']) - 1) * 100:.2f}%")
        print(f"매도 수량: {sell_volume:.8f} BTC")

        if TEST_MODE:
            # 테스트 모드 - 실제 주문 실행 안 함
            print("[테스트 모드] 실제 매도 주문은 실행되지 않았습니다.")
            order = {
                'uuid': f'dummy-uuid-{uuid.uuid4()}',
                'side': 'ask',
                'ord_type': 'market',
                'price': 0,
                'state': 'done',
                'market': TICKER,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'volume': sell_volume,
                'remaining_volume': 0,
                'reserved_fee': 0,
                'remaining_fee': 0,
                'paid_fee': 0,
                'locked': sell_volume,
                'executed_volume': sell_volume,
                'trades_count': 1
            }
        else:
            # 실제 시장가 매도 주문
            order = upbit.sell_market_order(TICKER, sell_volume)

            # 주문 상태 확인을 위한 대기
            time.sleep(1)
            order_info = upbit.get_order(order['uuid'])
            if order_info and 'state' in order_info:
                print(f"주문 상태: {order_info['state']}")
                order = order_info

        # 익절 상태 업데이트
        buy_record["profit_taken"] = True

        print(f"{buy_record['order']}차 매수분 익절 완료")
        print(f"예상 수익: {(current_price - buy_record['price']) * sell_volume:,.0f}원")

        return order

    except Exception as e:
        print(f"매도 주문 중 오류 발생: {e}")
        return None


def sell_all_bitcoin():
    """보유한 모든 비트코인을 시장가로 판매하는 함수"""
    try:
        # 보유 비트코인 확인
        btc_balance = upbit.get_balance(TICKER)
        if btc_balance is None or btc_balance <= 0:
            print("판매할 비트코인이 없습니다.")
            return False

        print(f"\n===== 비트코인 전량 매도 =====")
        print(f"매도 수량: {btc_balance:.8f} BTC")

        # 현재 가격 조회
        current_price = get_current_price(TICKER)
        if current_price is not None:
            estimated_value = btc_balance * current_price
            print(f"예상 매도 금액: {estimated_value:,.0f}원")

        if TEST_MODE:
            # 테스트 모드 - 실제 주문 실행 안 함
            print("[테스트 모드] 실제 매도 주문은 실행되지 않았습니다.")
            order = {
                'uuid': f'dummy-uuid-{uuid.uuid4()}',
                'side': 'ask',
                'ord_type': 'market',
                'price': 0,
                'state': 'done',
                'market': TICKER,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'volume': btc_balance,
                'remaining_volume': 0,
                'reserved_fee': 0,
                'remaining_fee': 0,
                'paid_fee': 0,
                'locked': btc_balance,
                'executed_volume': btc_balance,
                'trades_count': 1
            }
        else:
            # 실제 시장가 매도 주문
            order = upbit.sell_market_order(TICKER, btc_balance)

            # 주문 상태 확인을 위한 대기
            time.sleep(1)
            order_info = upbit.get_order(order['uuid'])
            if order_info and 'state' in order_info:
                print(f"주문 상태: {order_info['state']}")

        print("==== 매도 주문 완료 ====")
        print(f"주문 ID: {order['uuid']}")

        # 모든 매수 기록 초기화
        buy_records.clear()

        # 최종 잔액 조회
        time.sleep(1)
        final_balance = get_account_balance()
        print(f"\n===== 최종 계정 상태 =====")
        print(f"보유 원화: {final_balance['krw_balance']:,.0f}원")
        print(f"보유 비트코인: {final_balance['btc_balance']:.8f} BTC")

        return True

    except Exception as e:
        print(f"매도 주문 중 오류 발생: {e}")
        return False


def monitor_prices():
    """가격을 모니터링하여 추가 매수 및 익절 조건을 확인하는 함수"""
    global monitoring_active

    print(f"\n===== 가격 모니터링 시작 =====")
    print(f"모니터링 간격: {MONITOR_INTERVAL}초")
    print(f"추가 매수 조건: 이전 매수 대비 {PRICE_DROP_PERCENT}% 하락")
    print(f"익절 조건: 매수 가격 대비 {PROFIT_PERCENT}% 상승")

    while monitoring_active:
        try:
            # 현재 가격 조회
            current_price = get_current_price(TICKER)
            if current_price is None:
                print("현재 가격을 조회할 수 없습니다. 다음 모니터링까지 대기...")
                time.sleep(MONITOR_INTERVAL)
                continue

            # 현재 시간과 가격 출력
            now = datetime.now()
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] 현재 가격: {current_price:,.0f}원")

            # 원화 잔액 확인 (추가 매수 가능 여부)
            balance = get_account_balance()

            # 1. 익절 조건 확인 (매수 가격보다 3% 상승)
            for buy_record in buy_records:
                if not buy_record["profit_taken"] and current_price >= buy_record["profit_price"]:
                    print(f"{buy_record['order']}차 매수분 익절 조건 충족!")
                    execute_sell(buy_record)

            # 2. 추가 매수 조건 확인 (이전 매수 대비 5% 하락)
            if len(buy_records) > 0 and len(buy_records) < SPLIT_COUNT:
                last_buy = buy_records[-1]
                next_buy_price = last_buy["price"] * (1 - PRICE_DROP_PERCENT / 100)

                if current_price <= next_buy_price:
                    order_num = len(buy_records) + 1
                    print(f"{order_num}차 매수 조건 충족! (이전 매수 가격 대비 {PRICE_DROP_PERCENT}% 하락)")

                    # 첫 매수와 동일한 금액으로 추가 매수
                    buy_amount = buy_records[0]["amount"]

                    # 원화 잔액 확인
                    if balance["krw_balance"] < buy_amount:
                        print(f"원화 잔액 부족: {balance['krw_balance']:,.0f}원 / 필요: {buy_amount:,.0f}원")
                    else:
                        execute_buy(order_num, buy_amount)

            # 모든 매수가 익절되었는지 확인
            all_profit_taken = all(record["profit_taken"] for record in buy_records)
            if len(buy_records) == SPLIT_COUNT and all_profit_taken:
                print("모든 분할 매수가 익절되었습니다! 전략 완료.")
                monitoring_active = False
                break

            # 다음 모니터링까지 대기
            time.sleep(MONITOR_INTERVAL)

        except Exception as e:
            print(f"모니터링 중 오류 발생: {e}")
            time.sleep(MONITOR_INTERVAL)


def start_monitoring():
    """가격 모니터링 스레드를 시작하는 함수"""
    global monitoring_active, monitoring_thread

    if monitoring_active:
        print("이미 모니터링이 진행 중입니다.")
        return False

    if len(buy_records) == 0:
        print("먼저 첫 매수를 등록해야 모니터링을 시작할 수 있습니다.")
        return False

    monitoring_active = True
    monitoring_thread = threading.Thread(target=monitor_prices)
    monitoring_thread.daemon = True
    monitoring_thread.start()

    return True


def stop_monitoring():
    """가격 모니터링 스레드를 중지하는 함수"""
    global monitoring_active

    if not monitoring_active:
        print("현재 모니터링이 진행 중이 아닙니다.")
        return False

    monitoring_active = False
    print("모니터링을 중지합니다...")
    time.sleep(1)

    return True


def print_buy_records():
    """현재까지의 매수 기록을 출력하는 함수"""
    if len(buy_records) == 0:
        print("매수 기록이 없습니다.")
        return

    print("\n===== 매수 기록 =====")
    print(f"{'차수':^5} | {'매수가격':^12} | {'매수시간':^19} | {'매수수량':^12} | {'매수금액':^12} | {'익절가격':^12} | {'익절여부':^8}")
    print("-" * 90)

    total_amount = 0
    total_volume = 0

    for record in buy_records:
        time_str = record["time"].strftime('%Y-%m-%d %H:%M:%S')
        print(
            f"{record['order']:^5} | {record['price']:,.0f} | {time_str} | {record['volume']:.8f} | {record['amount']:,.0f} | {record['profit_price']:,.0f} | {'완료' if record['profit_taken'] else '대기'}")

        total_amount += record["amount"]
        total_volume += record["volume"]

    current_price = get_current_price(TICKER)
    if current_price:
        current_value = total_volume * current_price
        profit_loss = current_value - total_amount
        profit_percent = (profit_loss / total_amount) * 100 if total_amount > 0 else 0

        print("-" * 90)
        print(f"총 투자금액: {total_amount:,.0f}원")
        print(f"총 매수수량: {total_volume:.8f} BTC")
        print(f"현재 평가금액: {current_value:,.0f}원")
        print(f"현재 수익률: {profit_percent:.2f}% ({profit_loss:,.0f}원)")


def reset_strategy():
    """전략을 초기화하는 함수"""
    global buy_records, monitoring_active

    if monitoring_active:
        stop_monitoring()

    buy_records.clear()
    print("전략이 초기화되었습니다. 새로운 세븐스플릿을 시작할 수 있습니다.")


# 메인 실행 부분
if __name__ == "__main__":
    # API 키 검증
    if not access_key or not secret_key:
        print("===== 오류 =====")
        print("업비트 API 키가 설정되지 않았습니다.")
        print(".env 파일에 UPBIT_ACCESS_KEY와 UPBIT_SECRET_KEY를 올바르게 설정해주세요.")
        exit(1)

    # API 연결 테스트
    try:
        # 간단한 API 호출로 연결 테스트
        balance_test = upbit.get_balances()
        if balance_test is None:
            print("===== 오류 =====")
            print("업비트 서버에 연결할 수 없거나 인증에 실패했습니다.")
            print("API 키가 올바른지 확인해주세요.")
            exit(1)
    except Exception as e:
        print(f"===== 오류 =====")
        print(f"업비트 API 연결 중 오류가 발생했습니다: {e}")
        print("인터넷 연결 및 API 키를 확인해주세요.")
        exit(1)

    print("업비트 API 연결 성공!")

    if TEST_MODE:
        print("\n===== 테스트 모드 활성화 =====")
        print("실제 매수/매도 주문은 실행되지 않습니다.")

    while True:
        try:
            # 계정 정보 출력
            print("\n===== 계정 정보 =====")
            balance = get_account_balance()
            print(f"보유 원화: {balance['krw_balance']:,.0f}원")
            print(f"보유 비트코인: {balance['btc_balance']:.8f} BTC")
            if balance['btc_balance'] > 0 and balance['current_price']:
                print(f"비트코인 평가금액: {balance['btc_value']:,.0f}원")
                print(f"총 평가금액: {balance['total_value']:,.0f}원")

            # 현재 전략 상태 출력
            if len(buy_records) > 0:
                completed_count = sum(1 for r in buy_records if r["profit_taken"])
                print(f"\n===== 세븐스플릿 진행 상황 =====")
                print(f"총 매수 횟수: {len(buy_records)}/{SPLIT_COUNT}")
                print(f"익절 완료: {completed_count}/{len(buy_records)}")

                if monitoring_active:
                    print("모니터링 상태: 활성화")
                else:
                    print("모니터링 상태: 비활성화")

            # 사용자 입력 받기
            print("\n===== 세븐스플릿 전략 메뉴 =====")
            print("1. 매수/익절 모니터링 시작")
            print("2. 모니터링 중지")
            print("3. 매수 기록 확인")
            print("4. 보유 비트코인 전량 매도")
            print("5. 전략 초기화")
            print("6. 종료")
            choice = input("\n원하는 기능을 선택하세요 (1-6): ")

            if choice == '1':
                # 매수 기록이 없으면 자동으로 첫 매수 감지
                if len(buy_records) == 0:
                    if detect_first_buy():
                        start_monitoring()
                    else:
                        print("첫 매수를 감지할 수 없습니다. 업비트 앱에서 비트코인을 먼저 구매해주세요.")
                else:
                    start_monitoring()
            elif choice == '2':
                stop_monitoring()
            elif choice == '3':
                print_buy_records()
            elif choice == '4':
                sell_all_bitcoin()
            elif choice == '5':
                reset_strategy()
            elif choice == '6':
                if monitoring_active:
                    stop_monitoring()
                print("프로그램을 종료합니다.")
                break
            else:
                print("올바른 옵션을 선택해주세요 (1-6)")

            # 각 작업 후 잠시 대기
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n사용자에 의해 프로그램이 중단되었습니다.")
            if monitoring_active:
                stop_monitoring()
            break
        except Exception as e:
            print(f"\n===== 오류 발생 =====")
            print(f"오류 내용: {e}")
            print("프로그램을 계속 실행하시겠습니까?")
            cont = input("계속하려면 y, 종료하려면 아무 키나 누르세요: ")
            if cont.lower() != 'y':
                if monitoring_active:
                    stop_monitoring()
                print("프로그램을 종료합니다.")
                break
            print("프로그램을 계속 실행합니다.")