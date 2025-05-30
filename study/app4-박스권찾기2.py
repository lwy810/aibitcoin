# streamlit run app4-박스권찾기2.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import ccxt
import numpy as np
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

st.set_page_config(page_title="암호화폐 캔들 차트 - 바이낸스", layout="wide")

# 제목
st.title("📈 암호화폐 캔들 차트 - 바이낸스")

# ──────────────────────────────
# 📥 바이낸스 데이터 로드 (5년간)
@st.cache_data(ttl=1800, show_spinner=False)  # 30분마다 캐시 갱신
def load_binance_data(symbol='BTC/USDT', days=1825):
    """바이낸스에서 암호화폐 데이터 로드"""
    try:
        # 바이낸스 거래소 객체 생성
        exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
            'sandbox': False,  # 실제 데이터 사용
        })
        
        st.info(f"🔍 최신 {days}일 데이터를 바이낸스에서 가져오는 중...")
        
        # 최신 데이터부터 역순으로 가져오기 (since 파라미터 없이)
        # limit을 지정해서 원하는 일수만큼 가져오기
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=days)
        
        if not ohlcv:
            st.warning("❌ 바이낸스에서 데이터를 받지 못했습니다.")
            return None
        
        st.success(f"✅ 바이낸스에서 {len(ohlcv)}개 데이터 포인트를 받았습니다.")
        
        # DataFrame으로 변환
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # timestamp를 날짜로 변환 (UTC 기준)
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # 한국 시간으로 변환
        df['Date'] = df['Date'].dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
        
        df = df.drop('timestamp', axis=1)
        
        # 날짜 순으로 정렬 (최신이 마지막에 오도록)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # 디버그: 실제 받은 데이터 범위
        actual_start = df['Date'].min()
        actual_end = df['Date'].max()
        st.info(f"📅 실제 받은 데이터: {actual_start.strftime('%Y-%m-%d')} ~ {actual_end.strftime('%Y-%m-%d')} ({len(df)}일)")
        
        # 한국 시간 기준으로 오늘 날짜 계산
        kst_now = datetime.now()
        today = kst_now.date()
        
        # 최신 데이터 날짜 확인
        latest_data_date = actual_end.date()
        days_behind = (today - latest_data_date).days
        
        if days_behind <= 1:
            st.success(f"🟢 최신 데이터입니다! (최신: {latest_data_date}, 현재: {today})")
        elif days_behind <= 3:
            st.warning(f"🟡 {days_behind}일 전 데이터입니다. (최신: {latest_data_date}, 현재: {today})")
        else:
            st.error(f"🔴 {days_behind}일 전 데이터입니다. (최신: {latest_data_date}, 현재: {today})")
        
        return df
        
    except Exception as e:
        st.error(f"바이낸스 데이터 로드 중 오류 발생: {e}")
        st.error(f"오류 상세: {str(e)}")
        return None

# ──────────────────────────────
# 사이드바 설정
st.sidebar.header("📊 차트 설정")

# 심볼 선택
st.sidebar.subheader("💰 거래 페어 선택")

# 카테고리별로 구분
crypto_pairs = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", 
    "ADA/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"
]

stablecoin_pairs = [
    "USDC/USDT", "DAI/USDT", "BUSD/USDT", "TUSD/USDT"
]

# 라디오 버튼으로 카테고리 선택
category = st.sidebar.radio(
    "카테고리",
    ["🚀 암호화폐", "🏦 스테이블코인"]
)

if category == "🚀 암호화폐":
    selected_symbol = st.sidebar.selectbox(
        "암호화폐 선택",
        crypto_pairs,
        index=0
    )
    st.sidebar.info("💡 암호화폐의 USDT 기준 가격을 확인할 수 있습니다.")
else:
    selected_symbol = st.sidebar.selectbox(
        "스테이블코인 페어 선택", 
        stablecoin_pairs,
        index=0
    )
    st.sidebar.info("💡 스테이블코인 간의 페그(peg) 상태를 확인할 수 있습니다.")

# 선택된 페어 정보
if "USDT" in selected_symbol:
    base_coin = selected_symbol.split("/")[0]
    if base_coin in ["USDC", "DAI", "BUSD", "TUSD"]:
        st.sidebar.success(f"📊 {base_coin} ↔ USDT 페어를 분석합니다.")
    else:
        st.sidebar.success(f"📊 {base_coin}의 USDT 가격을 분석합니다.")

# 기간 선택
period_options = {
    "전체 기간 (5년)": 1825,
    "최근 3년": 1095,
    "최근 2년": 730,
    "최근 1년": 365,
    "최근 6개월": 180,
    "최근 3개월": 90,
    "최근 1개월": 30
}

selected_period = st.sidebar.selectbox(
    "표시 기간 선택",
    list(period_options.keys()),
    index=0
)

# 이동평균선 설정
show_ma = st.sidebar.checkbox("이동평균선 표시", value=True)
if show_ma:
    ma_periods = st.sidebar.multiselect(
        "이동평균 기간",
        [5, 10, 20, 50, 100, 200],
        default=[20, 50, 200]
    )

# 거래량 표시 옵션
show_volume = st.sidebar.checkbox("거래량 표시", value=True)

# 볼린저 밴드 옵션
show_bollinger = st.sidebar.checkbox("볼린저 밴드 표시", value=False)
if show_bollinger:
    bb_period = st.sidebar.slider("볼린저 밴드 기간", 10, 50, 20)
    bb_std = st.sidebar.slider("표준편차 배수", 1.0, 3.0, 2.0, 0.1)

# 데이터 새로고침 버튼
st.sidebar.markdown("---")
if st.sidebar.button("🔄 최신 데이터 새로고침"):
    st.cache_data.clear()
    st.rerun()

# ──────────────────────────────
# 데이터 로드
with st.spinner(f"📊 바이낸스에서 {selected_symbol} 데이터를 불러오는 중..."):
    raw_data = load_binance_data(selected_symbol, period_options[selected_period])

if raw_data is None:
    st.error("❌ 데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.")
    st.stop()

# 선택된 기간만큼 데이터 필터링
days_to_show = period_options[selected_period]
df = raw_data.tail(days_to_show).copy()

# ──────────────────────────────
# 기술적 지표 계산
if show_ma and ma_periods:
    for period in ma_periods:
        df[f'MA_{period}'] = df['close'].rolling(window=period).mean()

if show_bollinger:
    df['BB_SMA'] = df['close'].rolling(window=bb_period).mean()
    df['BB_STD'] = df['close'].rolling(window=bb_period).std()
    df['BB_Upper'] = df['BB_SMA'] + (bb_std * df['BB_STD'])
    df['BB_Lower'] = df['BB_SMA'] - (bb_std * df['BB_STD'])

# ──────────────────────────────
# 정보 표시
col1, col2, col3, col4 = st.columns(4)

with col1:
    current_price = df['close'].iloc[-1]
    # 가격에 따라 소수점 자리수 조정
    if current_price >= 1000:
        price_format = f"{current_price:,.2f} USDT"
    elif current_price >= 1:
        price_format = f"{current_price:.4f} USDT"
    else:
        price_format = f"{current_price:.6f} USDT"
    st.metric("현재가", price_format)

with col2:
    price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
    change_pct = (price_change / df['close'].iloc[-2]) * 100
    if abs(price_change) >= 1000:
        change_format = f"{price_change:+,.2f} USDT"
    elif abs(price_change) >= 1:
        change_format = f"{price_change:+.4f} USDT"
    else:
        change_format = f"{price_change:+.6f} USDT"
    st.metric("전일 대비", change_format, f"{change_pct:+.2f}%")

with col3:
    max_price = df['high'].max()
    if max_price >= 1000:
        max_format = f"{max_price:,.2f} USDT"
    elif max_price >= 1:
        max_format = f"{max_price:.4f} USDT"
    else:
        max_format = f"{max_price:.6f} USDT"
    st.metric(f"{selected_period} 최고가", max_format)

with col4:
    min_price = df['low'].min()
    if min_price >= 1000:
        min_format = f"{min_price:,.2f} USDT"
    elif min_price >= 1:
        min_format = f"{min_price:.4f} USDT"
    else:
        min_format = f"{min_price:.6f} USDT"
    st.metric(f"{selected_period} 최저가", min_format)

# 데이터 기간 정보
start_date = df['Date'].min()
end_date = df['Date'].max()
data_days = len(df)
current_time = datetime.now()

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.info(f"📅 데이터 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({data_days}일)")

with col_info2:
    # 최신 데이터가 얼마나 최근인지 계산
    time_diff = current_time - end_date.to_pydatetime()
    hours_diff = time_diff.total_seconds() / 3600
    
    if hours_diff < 24:
        freshness = f"🟢 최신 ({hours_diff:.1f}시간 전)"
    elif hours_diff < 48:
        freshness = f"🟡 어제 데이터 ({hours_diff:.0f}시간 전)"
    else:
        freshness = f"🔴 {hours_diff/24:.0f}일 전 데이터"
    
    st.info(f"🏛️ 바이낸스 {selected_symbol} - {freshness}")

# 현재 시간 표시
st.caption(f"현재 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')} KST | 최신 데이터: {end_date.strftime('%Y-%m-%d')} KST")

# ──────────────────────────────
# 📈 캔들 차트 생성
if show_volume:
    # 서브플롯 생성 (가격 + 거래량)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('가격', '거래량'),
        row_heights=[0.75, 0.25]  # 3:1 비율 (가격 75%, 거래량 25%)
    )
    
    # 캔들스틱 차트
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=selected_symbol,
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4444"
        ),
        row=1, col=1
    )
    
    # 거래량 바 차트
    colors = ['#00ff88' if close >= open else '#ff4444' 
              for close, open in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['volume'],
            name="거래량",
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
else:
    # 단일 차트 (가격만)
    fig = go.Figure()
    
    # 캔들스틱 차트
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=selected_symbol,
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4444"
        )
    )

# ──────────────────────────────
# 이동평균선 추가
if show_ma and ma_periods:
    colors = ['#ffff00', '#ff8800', '#8800ff', '#00ffff', '#ff00ff', '#88ff00']
    for i, period in enumerate(ma_periods):
        if f'MA_{period}' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df[f'MA_{period}'],
                    mode='lines',
                    name=f'MA{period}',
                    line=dict(color=colors[i % len(colors)], width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )

# ──────────────────────────────
# 볼린저 밴드 추가
if show_bollinger and 'BB_Upper' in df.columns:
    # 상단 밴드
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['BB_Upper'],
            mode='lines',
            name=f'BB Upper({bb_period})',
            line=dict(color='rgba(173,216,230,0.8)', width=1),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # 하단 밴드
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['BB_Lower'],
            mode='lines',
            name=f'BB Lower({bb_period})',
            line=dict(color='rgba(173,216,230,0.8)', width=1),
            fill='tonexty',  # 상단 밴드와 채우기
            fillcolor='rgba(173,216,230,0.1)',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # 중간선 (SMA)
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['BB_SMA'],
            mode='lines',
            name=f'BB SMA({bb_period})',
            line=dict(color='rgba(173,216,230,1)', width=1),
            opacity=0.9
        ),
        row=1, col=1
    )

# ──────────────────────────────
# 차트 레이아웃 설정
currency_name = "USD" if "USD" in selected_symbol else "EUR" if "EUR" in selected_symbol else "BTC" if "BTC" in selected_symbol else "ETH"

fig.update_layout(
    title=f"바이낸스 {selected_symbol} 캔들 차트 - {selected_period}",
    xaxis_title="날짜",
    yaxis_title="가격 (USDT)",
    xaxis_rangeslider_visible=False,
    plot_bgcolor="#1e1e1e",
    paper_bgcolor="#1e1e1e",
    font=dict(color="white"),
    height=800 if show_volume else 600,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Y축 설정
fig.update_yaxes(title_text="가격 (USDT)", row=1, col=1)
if show_volume:
    fig.update_yaxes(title_text="거래량", row=2, col=1)

# ──────────────────────────────
# 차트 표시
st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────
# 최근 데이터 테이블
st.subheader("📊 최근 10일 데이터")
recent_data = df.tail(10)[['Date', 'open', 'high', 'low', 'close', 'volume']].copy()

# 포맷팅 (소수점 4자리)
recent_data['open'] = recent_data['open'].round(4)
recent_data['high'] = recent_data['high'].round(4)
recent_data['low'] = recent_data['low'].round(4)
recent_data['close'] = recent_data['close'].round(4)
recent_data['volume'] = recent_data['volume'].round(0).astype(int)

# 컬럼명 한글로 변경
recent_data.columns = ['날짜', '시가', '고가', '저가', '종가', '거래량']
recent_data = recent_data.sort_values('날짜', ascending=False)

st.dataframe(recent_data, use_container_width=True)

# ──────────────────────────────
# 통계 정보
st.subheader("📈 기간별 통계")
col1, col2 = st.columns(2)

with col1:
    st.write("**가격 통계**")
    
    # 가격 포맷팅 함수
    def format_price(price):
        if price >= 1000:
            return f"{price:,.2f} USDT"
        elif price >= 1:
            return f"{price:.4f} USDT"
        else:
            return f"{price:.6f} USDT"
    
    price_stats = pd.DataFrame({
        '항목': ['평균가', '최고가', '최저가', '표준편차', '변동률(%)'],
        '값': [
            format_price(df['close'].mean()),
            format_price(df['high'].max()), 
            format_price(df['low'].min()),
            format_price(df['close'].std()),
            f"{((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.2f}%"
        ]
    })
    st.dataframe(price_stats, hide_index=True)

with col2:
    st.write("**거래량 통계**") 
    volume_stats = pd.DataFrame({
        '항목': ['평균 거래량', '최대 거래량', '최소 거래량'],
        '값': [
            f"{df['volume'].mean():,.0f}",
            f"{df['volume'].max():,.0f}",
            f"{df['volume'].min():,.0f}"
        ]
    })
    st.dataframe(volume_stats, hide_index=True)
