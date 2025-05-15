# pip install pandas (터미널에서 실행)

# 코드에서 판다스 불러오기
import pandas as pd
import numpy as np  # 데이터 생성을 위해 NumPy도 함께 불러옵니다

# 4. CSV 파일에서 데이터프레임 생성 (예시)
# df4 = pd.read_csv('데이터.csv')
# print(df4)

# 샘플 데이터프레임 생성
df = pd.DataFrame({
    '이름': ['김철수', '이영희', '박민수', '최지은', '정다혜'],
    '나이': [25, 28, 22, 30, 35],
    '성별': ['남', '여', '남', '여', '여'],
    '직업': ['학생', '개발자', '디자이너', '교사', '의사'],
    '급여': [250, 420, 350, 380, 500]
})

# 샘플 데이터프레임 사용
print("원본 데이터프레임:")
print(df)
print("\n")

# 열 선택하기
print("'이름' 열 선택:")
print(df['이름'])  # 단일 열 선택
print("\n")

print("'이름'과 '직업' 열 선택:")
print(df[['이름', '직업']])  # 다중 열 선택
print("\n")

# loc로 행과 열 선택하기 (레이블 기반)
print("첫 번째 행 선택 (loc):")
print(df.loc[0])  # 첫 번째 행 선택
print("\n")

print("처음 3개 행의 '이름'과 '나이' 열 선택 (loc):")
print(df.loc[0:2, ['이름', '나이']])  # 처음 3개 행, 특정 열 선택
print("\n")

# iloc로 행과 열 선택하기 (위치 기반)
print("두 번째 행 선택 (iloc):")
print(df.iloc[1])  # 두 번째 행 선택
print("\n")

print("처음 3개 행, 2-3번째 열 선택 (iloc):")
print(df.iloc[0:3, 1:3])  # 처음 3개 행, 2-3번째 열 선택
print("\n")

# 조건을 이용한 선택
print("나이가 25 이상인 행:")
print(df[df['나이'] >= 25])
print("\n")

print("개발자나 의사인 행:")
print(df[df['직업'].isin(['개발자', '의사'])])
print("\n")

print("여성이면서 급여가 400 이상인 행:")
print(df[(df['성별'] == '여') & (df['급여'] >= 400)])
print("\n")