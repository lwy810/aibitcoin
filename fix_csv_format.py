import pandas as pd

# 원본 CSV 파일 읽기
print("원본 CSV 파일 읽는 중...")
df = pd.read_csv('btc_4h_data_2018_to_2025.csv')

print("원본 컬럼명:", df.columns.tolist())
print("원본 데이터 샘플:")
print(df.head())

# 컬럼명 매핑
column_mapping = {
    'Open time': 'timestamp',
    'Open': 'open',
    'High': 'high', 
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}

# 필요한 컬럼만 선택하고 이름 변경
required_columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']
df_processed = df[required_columns].copy()
df_processed = df_processed.rename(columns=column_mapping)

# timestamp 컬럼을 datetime으로 변환
df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])

print("\n처리된 컬럼명:", df_processed.columns.tolist())
print("처리된 데이터 샘플:")
print(df_processed.head())

# 새로운 CSV 파일로 저장
new_filename = 'btc_4h_data_2018_to_2025_fixed.csv'
df_processed.to_csv(new_filename, index=False)

print(f"\n✅ {new_filename} 파일이 생성되었습니다!")
print(f"📊 데이터 정보:")
print(f"   - 기간: {df_processed['timestamp'].min()} ~ {df_processed['timestamp'].max()}")
print(f"   - 총 레코드 수: {len(df_processed):,}개")
print(f"   - 컬럼: {list(df_processed.columns)}")

# 기존 파일명으로도 저장 (덮어쓰기)
df_processed.to_csv('btc_4h_data_2018_to_2025.csv', index=False)
print(f"✅ 기존 파일도 업데이트되었습니다!") 