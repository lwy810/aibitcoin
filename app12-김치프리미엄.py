import requests
import json
import time
from datetime import datetime

class KimchiPremiumCalculator:
    def __init__(self):
        self.upbit_url = "https://api.upbit.com/v1/ticker"
        self.binance_url = "https://api.binance.com/api/v3/ticker/price"
        
    def get_upbit_price(self, symbol):
        """업비트에서 가격 조회"""
        try:
            params = {'markets': f'KRW-{symbol}'}
            response = requests.get(self.upbit_url, params=params)
            data = response.json()
            if data:
                return float(data[0]['trade_price'])
            return None
        except Exception as e:
            print(f"업비트 API 오류: {e}")
            return None
    
    def get_binance_price(self, symbol):
        """바이낸스에서 가격 조회 (USDT 기준)"""
        try:
            params = {'symbol': f'{symbol}USDT'}
            response = requests.get(self.binance_url, params=params)
            data = response.json()
            return float(data['price'])
        except Exception as e:
            print(f"바이낸스 API 오류: {e}")
            return None
    
    def get_usd_krw_rate(self):
        """달러-원 환율 조회"""
        try:
            # 환율 API 사용 (예: exchangerate-api.com)
            url = "https://api.exchangerate-api.com/v4/latest/USD"
            response = requests.get(url)
            data = response.json()
            return data['rates']['KRW']
        except Exception as e:
            print(f"환율 API 오류: {e}")
            # 환율 API 실패시 대략적인 환율 사용
            return 1300  # 기본값
    
    def calculate_premium(self, symbol):
        """김치 프리미엄 계산"""
        print(f"\n=== {symbol} 김치 프리미엄 계산 ===")
        
        # 각 거래소 가격 조회
        upbit_price = self.get_upbit_price(symbol)
        binance_price = self.get_binance_price(symbol)
        usd_krw = self.get_usd_krw_rate()
        
        if upbit_price is None or binance_price is None:
            print("❌ 가격 조회 실패")
            return None
        
        # 바이낸스 가격을 원화로 환산
        binance_krw_price = binance_price * usd_krw
        
        # 김치 프리미엄 계산
        premium = ((upbit_price - binance_krw_price) / binance_krw_price) * 100
        
        # 결과 출력
        print(f"📊 현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💱 USD/KRW 환율: {usd_krw:,.2f}원")
        print(f"🇰🇷 업비트 가격: {upbit_price:,.0f}원")
        print(f"🌍 바이낸스 가격: ${binance_price:,.4f} ({binance_krw_price:,.0f}원)")
        print(f"📈 김치 프리미엄: {premium:+.2f}%")
        
        if premium > 0:
            print(f"💰 업비트가 {abs(premium):.2f}% 더 비쌉니다")
        else:
            print(f"💸 업비트가 {abs(premium):.2f}% 더 쌉니다")
        
        return {
            'symbol': symbol,
            'upbit_price': upbit_price,
            'binance_price': binance_price,
            'binance_krw_price': binance_krw_price,
            'usd_krw_rate': usd_krw,
            'premium': premium,
            'timestamp': datetime.now()
        }
    
    def monitor_multiple_coins(self, symbols, interval=30):
        """여러 코인의 김치 프리미엄 모니터링"""
        print("🚀 김치 프리미엄 모니터링 시작!")
        print(f"📊 대상 코인: {', '.join(symbols)}")
        print(f"⏰ 업데이트 간격: {interval}초")
        print("-" * 50)
        
        try:
            while True:
                results = []
                for symbol in symbols:
                    result = self.calculate_premium(symbol)
                    if result:
                        results.append(result)
                    time.sleep(1)  # API 호출 간격
                
                # 프리미엄 순으로 정렬
                if results:
                    results.sort(key=lambda x: x['premium'], reverse=True)
                    print(f"\n📊 프리미엄 랭킹:")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. {result['symbol']}: {result['premium']:+.2f}%")
                
                print(f"\n⏰ {interval}초 후 다시 조회...")
                print("=" * 70)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 모니터링을 종료합니다.")

# 사용 예시
if __name__ == "__main__":
    calculator = KimchiPremiumCalculator()
    
    # 주요 코인들
    major_coins = ['BTC', 'ETH', 'XRP', 'ADA', 'DOT', 'LINK', 'SOL', 'AVAX']
    
    print("🎯 김치 프리미엄 계산기")
    print("1. 단일 코인 조회")
    print("2. 여러 코인 모니터링")
    
    choice = input("\n선택하세요 (1 또는 2): ")
    
    if choice == '1':
        symbol = input("코인 심볼을 입력하세요 (예: BTC): ").upper()
        calculator.calculate_premium(symbol)
    
    elif choice == '2':
        print(f"기본 코인 목록: {', '.join(major_coins)}")
        custom_input = input("다른 코인을 추가하시겠습니까? (쉼표로 구분, 엔터는 건너뛰기): ")
        
        if custom_input.strip():
            custom_coins = [coin.strip().upper() for coin in custom_input.split(',')]
            major_coins.extend(custom_coins)
        
        interval = input("업데이트 간격(초, 기본값 30초): ")
        interval = int(interval) if interval.isdigit() else 30
        
        calculator.monitor_multiple_coins(major_coins, interval)
    
    else:
        print("잘못된 선택입니다.")