import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
import time
import os

# --- 데이터 수집 함수 (app13-AI 기반 ... .py 에서 가져옴) ---
def get_bitcoin_data(start_date='2018-01-01', end_date='2024-01-01'):
    """Yahoo Finance에서 비트코인(BTC-USD) 과거 데이터를 가져옵니다."""
    print("비트코인 데이터 가져오는 중...")
    df = yf.download('BTC-USD', start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    # 백테스팅을 위해 OHLC 데이터 모두 포함
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"비트코인 데이터 가져오기 완료. 형태: {df.shape}")
    return df

def get_google_trends_data(keyword='bitcoin', start_date='2018-01-01', end_date='2024-01-01', timeframe_months=3, delay_seconds=10):
    """주어진 키워드에 대한 Google 트렌드 데이터를 가져옵니다. 긴 기간은 분할하여 요청합니다."""
    print(f"'{keyword}'에 대한 Google 트렌드 데이터 가져오는 중 (시작: {start_date}, 종료: {end_date})...")
    pytrend = TrendReq(hl='en-US', tz=360)
    kw_list = [keyword]
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    all_trends_df = pd.DataFrame()
    current_start = start_dt
    while current_start < end_dt:
        current_end = current_start + pd.DateOffset(months=timeframe_months) - pd.DateOffset(days=1)
        if current_end > end_dt:
            current_end = end_dt
        timeframe_str = f'{current_start.strftime("%Y-%m-%d")} {current_end.strftime("%Y-%m-%d")}'
        print(f"  - Google Trends 요청: {timeframe_str}")
        try:
            pytrend.build_payload(kw_list, cat=0, timeframe=timeframe_str, geo='', gprop='')
            df_trends_partial = pytrend.interest_over_time()
            if not df_trends_partial.empty:
                if 'isPartial' in df_trends_partial.columns:
                    df_trends_partial = df_trends_partial.drop(columns=['isPartial'])
                if keyword in df_trends_partial.columns:
                    all_trends_df = pd.concat([all_trends_df, df_trends_partial[[keyword]]])
                else:
                    print(f"    경고: '{keyword}' 컬럼이 '{timeframe_str}' 기간의 트렌드 데이터에 없습니다.")
            else:
                print(f"    '{timeframe_str}' 기간에 대한 Google 트렌드 데이터가 없습니다.")
        except Exception as e:
            print(f"    Google Trends API 요청 중 오류 발생 ({timeframe_str}): {e}")
        current_start = current_end + pd.DateOffset(days=1)
        if current_start < end_dt:
            print(f"    {delay_seconds}초 대기...")
            time.sleep(delay_seconds)
    if not all_trends_df.empty:
        all_trends_df = all_trends_df.rename(columns={keyword: 'GoogleTrends'})
        all_trends_df = all_trends_df[~all_trends_df.index.duplicated(keep='first')]
        all_trends_df = all_trends_df.sort_index()
        all_trends_df['GoogleTrends'] = all_trends_df['GoogleTrends'].rolling(window=7, min_periods=1).mean()
        print(f"최종 Google 트렌드 데이터 가져오기 완료. 형태: {all_trends_df.shape}")
        return all_trends_df
    else:
        print("모든 Google 트렌드 데이터 요청이 실패했거나 데이터가 없습니다.")
        return pd.DataFrame()

# --- 기술 지표 계산 함수 (app13-AI 기반 ... .py 에서 가져옴) ---
def calculate_technical_indicators(df):
    """주어진 DataFrame에 대해 RSI 및 MACD를 계산합니다."""
    print("기술 지표 계산 중...")
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - macd_signal
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    print("기술 지표 계산 완료.")
    return df

def merge_ohlc_to_existing_csv():
    """기존 CSV 파일에 OHLC 데이터를 결합합니다."""
    csv_filename = 'bitcoin_processed_data.csv'
    
    print(f"기존 '{csv_filename}' 파일에 OHLC 데이터 결합 시작...")
    
    # 1. 기존 CSV 파일 읽기
    try:
        df_existing = pd.read_csv(csv_filename, index_col='Date', parse_dates=True)
        print(f"기존 CSV 파일 로드 완료. 형태: {df_existing.shape}")
        print(f"기존 컬럼: {df_existing.columns.tolist()}")
    except FileNotFoundError:
        print(f"기존 '{csv_filename}' 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"기존 CSV 파일 로드 중 오류: {e}")
        return
    
    # 2. 이미 OHLC 데이터가 있는지 확인
    missing_ohlc = [col for col in ['Open', 'High', 'Low'] if col not in df_existing.columns]
    if not missing_ohlc:
        print("기존 파일에 이미 OHLC 데이터가 모두 있습니다.")
        return
    
    print(f"누락된 OHLC 컬럼: {missing_ohlc}")
    
    # 3. 기존 데이터의 날짜 범위 확인
    start_date = df_existing.index.min().strftime('%Y-%m-%d')
    end_date = df_existing.index.max().strftime('%Y-%m-%d')
    print(f"기존 데이터 날짜 범위: {start_date} ~ {end_date}")
    
    # 4. 비트코인 OHLCV 데이터 새로 다운로드
    df_btc_ohlcv = get_bitcoin_data(start_date, end_date)
    print(f"새로 다운로드한 BTC OHLCV 데이터 형태: {df_btc_ohlcv.shape}")
    
    # 5. 기존 데이터와 OHLCV 데이터 병합
    print("데이터 병합 중...")
    
    # 기존 데이터에 누락된 OHLC 컬럼만 추가
    for col in missing_ohlc:
        if col in df_btc_ohlcv.columns:
            df_existing[col] = df_btc_ohlcv[col]
            print(f"'{col}' 컬럼 추가 완료")
        else:
            print(f"경고: '{col}' 컬럼이 새로 다운로드한 데이터에 없습니다.")
    
    # Volume도 업데이트 (더 정확한 데이터일 수 있음)
    if 'Volume' in df_btc_ohlcv.columns:
        df_existing['Volume'] = df_btc_ohlcv['Volume']
        print("'Volume' 컬럼 업데이트 완료")
    
    # 6. 결측치 처리
    print("결측치 처리 중...")
    df_existing = df_existing.ffill().bfill()
    df_existing.dropna(inplace=True)
    print(f"결측치 처리 후 데이터 형태: {df_existing.shape}")
    
    # 7. 업데이트된 CSV 파일 저장
    try:
        df_existing.to_csv(csv_filename, index=True)
        print(f"OHLC 데이터가 추가된 파일이 '{csv_filename}'으로 저장되었습니다.")
        print(f"최종 컬럼: {df_existing.columns.tolist()}")
        print(f"최종 데이터 처음 5행:\n{df_existing.head()}")
    except Exception as e:
        print(f"CSV 파일 저장 중 오류 발생: {e}")

def reorder_csv_columns():
    """CSV 파일의 컬럼 순서를 OHLC 표준 순서로 재정렬합니다."""
    csv_filename = 'bitcoin_processed_data.csv'
    
    print(f"'{csv_filename}' 파일의 컬럼 순서 재정렬 시작...")
    
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_filename, index_col='Date', parse_dates=True)
        print(f"기존 컬럼 순서: {df.columns.tolist()}")
        
        # 원하는 컬럼 순서 정의
        desired_order = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'GoogleTrends', 'RSI', 'MACD_Hist', 'SMA_20', 
            'Price_Change_Pct', 'NextDayPriceIncrease'
        ]
        
        # 모든 컬럼이 존재하는지 확인
        missing_cols = [col for col in desired_order if col not in df.columns]
        if missing_cols:
            print(f"경고: 다음 컬럼이 없습니다: {missing_cols}")
            # 존재하는 컬럼만으로 순서 조정
            desired_order = [col for col in desired_order if col in df.columns]
        
        # 컬럼 순서 재정렬
        df_reordered = df[desired_order]
        
        # 재정렬된 CSV 파일 저장
        df_reordered.to_csv(csv_filename, index=True)
        
        print(f"컬럼 순서가 재정렬되어 '{csv_filename}'에 저장되었습니다.")
        print(f"새로운 컬럼 순서: {df_reordered.columns.tolist()}")
        print(f"재정렬된 데이터 처음 5행:\n{df_reordered.head()}")
        
    except Exception as e:
        print(f"컬럼 순서 재정렬 중 오류 발생: {e}")

def main_get_csv():
    start_date = '2018-01-01'
    end_date = '2024-01-01'
    csv_filename = 'bitcoin_processed_data.csv'
    google_trends_timeframe_months = 3
    google_trends_delay_seconds = 10 # Google Trends API 요청 시 딜레이 (초)

    # 먼저 기존 파일이 있는지 확인하고, 있다면 OHLC 데이터만 결합
    if os.path.exists(csv_filename):
        print(f"기존 '{csv_filename}' 파일이 발견되었습니다.")
        user_choice = input("기존 파일에 OHLC 데이터를 결합하시겠습니까? (y/n): ").lower().strip()
        if user_choice in ['y', 'yes', '예', 'ㅇ']:
            merge_ohlc_to_existing_csv()
            return
        else:
            print("새로운 CSV 파일을 생성합니다.")

    print(f"데이터 수집 및 전처리 후 '{csv_filename}'으로 저장 시작...")

    # 1. 비트코인 데이터 가져오기
    df_btc = get_bitcoin_data(start_date, end_date)

    # 2. Google Trends 데이터 가져오기 (네트워크 오류 시 건너뛰기)
    try:
        df_trends = get_google_trends_data('bitcoin', start_date, end_date, 
                                           timeframe_months=google_trends_timeframe_months,
                                           delay_seconds=google_trends_delay_seconds)
    except Exception as e:
        print(f"Google Trends 데이터 수집 중 오류 발생: {e}")
        print("Google Trends 없이 비트코인 데이터만 사용하여 진행합니다.")
        df_trends = pd.DataFrame()

    # 3. 데이터 병합
    print("데이터 병합 중...")
    if df_trends.empty:
        print("Google Trends 데이터가 없어 BTC 데이터만 사용합니다.")
        df_merged = df_btc.copy()
        # Google Trends가 없을 경우를 대비해 컬럼 추가 (모두 0으로 채워짐)
        df_merged['GoogleTrends'] = 0 # NaN 대신 0으로 설정
    else:
        df_merged = pd.merge(df_btc, df_trends, left_index=True, right_index=True, how='left')
        # 병합 후 GoogleTrends 컬럼의 NaN을 0으로 채움
        df_merged['GoogleTrends'] = df_merged['GoogleTrends'].fillna(0)
    print(f"병합된 데이터 형태: {df_merged.shape}")

    # 4. 기술 지표 계산
    df_processed = calculate_technical_indicators(df_merged.copy()) # 원본 보존을 위해 복사본 사용

    # 5. 추가 특징 생성 (가격 변화율, 다음 날 가격 방향)
    print("추가 특징 생성 중...")
    df_processed['Price_Change_Pct'] = df_processed['Close'].pct_change() * 100
    df_processed['NextDayPriceIncrease'] = (df_processed['Close'].shift(-1) > df_processed['Close']).astype(int)
    print("추가 특징 생성 완료.")

    # 6. 결측치 처리 (기술 지표 계산 및 shift로 인해 발생 가능)
    print("결측치 처리 중 (ffill, bfill)...")
    df_processed = df_processed.ffill()
    df_processed = df_processed.bfill()
    # NextDayPriceIncrease로 인해 마지막 행에 NaN이 있을 수 있으므로 dropna 처리
    df_processed.dropna(subset=['NextDayPriceIncrease'], inplace=True)
    # RSI 등 초기값이 NaN일 수 있는 다른 컬럼들도 한번 더 확인 (보통 bfill로 해결됨)
    df_processed.dropna(inplace=True) 
    print(f"결측치 처리 후 데이터 형태: {df_processed.shape}")

    # 7. CSV 파일로 저장
    try:
        df_processed.to_csv(csv_filename, index=True) # 날짜 인덱스 포함하여 저장
        print(f"데이터가 성공적으로 '{csv_filename}' 파일로 저장되었습니다.")
        print(f"저장된 데이터 컬럼: {df_processed.columns.tolist()}")
        print(f"저장된 데이터 처음 5행:\n{df_processed.head()}")
    except Exception as e:
        print(f"CSV 파일 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    main_get_csv()
