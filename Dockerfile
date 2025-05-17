# 베이스 이미지로 파이썬 공식 이미지 사용
FROM python:3.9

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 환경 변수 설정 (실행 시 오버라이드 가능)
ENV UPBIT_ACCESS_KEY=""
ENV UPBIT_SECRET_KEY=""
ENV DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/137328575097500409/6qmjgQBt2YvNep1fYYEo5G8m0N-h3bvr7WpbDDJ-MsSwczj1-UV9kpOmP5DFU_ERZCVb"

# 컨테이너 실행 시 실행할 명령어
CMD ["python", "app.py"]