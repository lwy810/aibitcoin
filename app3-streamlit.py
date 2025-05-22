import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import time
from streamlit_autorefresh import st_autorefresh  # 자동 새로고침 추가

# 페이지 설정
st.set_page_config(
    page_title="업비트 그리드 트레이딩 대시보드",
    page_icon="📈",
    layout="wide"
)

# 데이터베이스 연결
def get_db_connection():
    return sqlite3.connect('trading_history.db')

# 데이터 로드 함수들
def load_trades(days=7):
    conn = get_db_connection()
    query = f"""
    SELECT * FROM trades 
    WHERE timestamp >= datetime('now', '-{days} days')
    ORDER BY timestamp DESC
    """
    trades_df = pd.read_sql_query(query, conn)
    conn.close()
    return trades_df

def load_balance_history(days=7):
    conn = get_db_connection()
    query = f"""
    SELECT * FROM balance_history 
    WHERE timestamp >= datetime('now', '-{days} days')
    ORDER BY timestamp ASC
    """
    balance_df = pd.read_sql_query(query, conn)
    conn.close()
    return balance_df

def get_summary_stats():
    conn = get_db_connection()
    
    # 전체 거래 통계
    trades_query = """
    SELECT 
        COUNT(*) as total_trades,
        SUM(CASE WHEN type = 'buy' THEN 1 ELSE 0 END) as buy_count,
        SUM(CASE WHEN type = 'sell' THEN 1 ELSE 0 END) as sell_count,
        SUM(CASE WHEN type = 'buy' THEN amount ELSE 0 END) as total_buy_amount,
        SUM(CASE WHEN type = 'sell' THEN amount ELSE 0 END) as total_sell_amount,
        SUM(fee) as total_fees,
        SUM(profit) as total_profit
    FROM trades
    """
    
    # 최근 잔고 정보
    balance_query = """
    SELECT * FROM balance_history 
    ORDER BY timestamp DESC LIMIT 1
    """
    
    trades_stats = pd.read_sql_query(trades_query, conn)
    latest_balance = pd.read_sql_query(balance_query, conn)
    
    conn.close()
    return trades_stats, latest_balance

def load_grid_status():
    """현재 그리드 상태를 가져옵니다."""
    conn = get_db_connection()
    query = """
    WITH latest_grid AS (
        SELECT 
            grid_level,
            buy_price_target,
            sell_price_target,
            order_krw_amount,
            is_bought,
            actual_bought_volume,
            actual_buy_fill_price,
            timestamp,
            ROW_NUMBER() OVER (PARTITION BY grid_level ORDER BY timestamp DESC) as rn
        FROM grid 
        WHERE timestamp >= datetime('now', '-1 day')
    )
    SELECT 
        grid_level,
        buy_price_target,
        sell_price_target,
        order_krw_amount,
        is_bought,
        actual_bought_volume,
        actual_buy_fill_price,
        timestamp
    FROM latest_grid 
    WHERE rn = 1
    ORDER BY grid_level ASC
    """
    grid_df = pd.read_sql_query(query, conn)
    conn.close()
    return grid_df

def get_coin_name(ticker):
    coin_names = {
        "KRW-XRP": "리플",
        "KRW-BTC": "비트코인",
        "KRW-ETH": "이더리움",
        # 필요시 추가
    }
    return coin_names.get(ticker, ticker)

def get_latest_price():
    conn = get_db_connection()
    query = """
    SELECT current_price FROM balance_history ORDER BY timestamp DESC LIMIT 1
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    if not df.empty:
        return df['current_price'].iloc[0]
    return None

# 메인 대시보드
def main():
    st_autorefresh(interval=10 * 1000, key="refresh")  # 10초마다 새로고침
    st.title("📈 업비트 그리드 트레이딩 대시보드")
    
    # 자동 새로고침 상태 표시
    refresh_status = st.empty()
    # 마지막 업데이트 시간 표시
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    refresh_status.info(f"마지막 업데이트: {current_time} (10초마다 자동 갱신)")
    
    # 컨테이너 생성
    metrics_container = st.empty()
    grid_container = st.empty()
    trades_container = st.empty()
    
    # 상단에 추가 (main 함수 내)
    TICKER = "KRW-XRP"  # 실제 환경에 맞게 변수명 맞춰주세요
    PRICE_CHANGE = 4    # 실제 환경에 맞게 상수 사용

    # 현재가 가져오기 (이미 get_summary_stats 등에서 사용 중이면 재활용)
    current_price = None
    try:
        # balance_history에서 최근 current_price 사용
        _, latest_balance = get_summary_stats()
        if not latest_balance.empty:
            current_price = latest_balance['current_price'].iloc[0]
    except Exception:
        current_price = None

    with metrics_container.container():
        # 코인명/현재가 출력 (메트릭 위로 이동)
        coin_name = get_coin_name(TICKER)
        if current_price is not None:
            st.markdown(f"### {TICKER} ({coin_name}) | 현재가: **{current_price:,.2f}원**")
        else:
            st.markdown(f"### {TICKER} ({coin_name}) | 현재가: -")
        # 요약 통계 (7일 고정)
        trades_stats, latest_balance = get_summary_stats()
        
        # 상단 메트릭
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "총 거래 횟수",
                f"{trades_stats['total_trades'].iloc[0]:,}회",
                f"매수: {trades_stats['buy_count'].iloc[0]:,}회 / 매도: {trades_stats['sell_count'].iloc[0]:,}회"
            )
        
        with col2:
            total_profit = trades_stats['total_profit'].iloc[0]
            profit_color = "normal" if total_profit >= 0 else "inverse"
            st.metric(
                "총 수익",
                f"{total_profit:,.0f}원",
                f"수수료: {trades_stats['total_fees'].iloc[0]:,.0f}원",
                delta_color=profit_color
            )
        
        with col3:
            if not latest_balance.empty:
                # 이전 자산과 비교하여 변화율 계산
                balance_df = load_balance_history(7)
                if not balance_df.empty and len(balance_df) > 1:
                    prev_assets = balance_df['total_assets'].iloc[-2]
                    current_assets = latest_balance['total_assets'].iloc[0]
                    assets_change = current_assets - prev_assets
                    assets_change_pct = (assets_change / prev_assets) * 100 if prev_assets > 0 else 0
                    delta_text = f"{assets_change:+,.0f}원 ({assets_change_pct:+.2f}%)"
                else:
                    delta_text = "변화 없음"

                st.metric(
                    "현재 총 자산",
                    f"{latest_balance['total_assets'].iloc[0]:,.0f}원",
                    delta_text
                )
        
        with col4:
            if not latest_balance.empty:
                coin_value = latest_balance['coin_balance'].iloc[0] * latest_balance['current_price'].iloc[0]
                # 이전 코인 가치와 비교
                if not balance_df.empty and len(balance_df) > 1:
                    prev_coin_value = balance_df['coin_balance'].iloc[-2] * balance_df['current_price'].iloc[-2]
                    coin_value_change = coin_value - prev_coin_value
                    coin_value_change_pct = (coin_value_change / prev_coin_value) * 100 if prev_coin_value > 0 else 0
                    delta_text = f"{coin_value_change:+,.0f}원 ({coin_value_change_pct:+.2f}%)"
                else:
                    delta_text = "변화 없음"

                st.metric(
                    "보유 코인 가치",
                    f"{coin_value:,.0f}원",
                    delta_text
                )
    
    with grid_container.container():
        # 그리드 현황
        st.subheader("그리드 현황")
        grid_df = load_grid_status()
        
        if not grid_df.empty:
            # 컬럼명 한글로 변경
            grid_df = grid_df.rename(columns={
                'grid_level': '구간',
                'buy_price_target': '매수목표가',
                'sell_price_target': '매도목표가',
                'order_krw_amount': '주문금액',
                'is_bought': '매수상태',
                'actual_bought_volume': '매수수량',
                'actual_buy_fill_price': '매수가격',
                'timestamp': '최종업데이트'
            })
            
            # 데이터 포맷팅
            grid_df['매수목표가'] = grid_df['매수목표가'].apply(lambda x: f"{x:,.2f}원")
            grid_df['매도목표가'] = grid_df['매도목표가'].apply(lambda x: f"{x:,.2f}원")
            grid_df['주문금액'] = grid_df['주문금액'].apply(lambda x: f"{x:,.0f}원")
            grid_df['매수수량'] = grid_df['매수수량'].apply(lambda x: f"{x:.8f}" if x > 0 else "-")
            grid_df['매수가격'] = grid_df['매수가격'].apply(lambda x: f"{x:,.2f}원" if x > 0 else "-")
            grid_df['매수상태'] = grid_df['매수상태'].apply(lambda x: "매수완료" if x else "대기중")
            grid_df['최종업데이트'] = pd.to_datetime(grid_df['최종업데이트']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 구간 컬럼에 화살표 추가
            def add_arrow_to_current_grid(row):
                try:
                    price = current_price
                    buy_target = float(str(row['매수목표가']).replace('원','').replace(',',''))
                    if buy_target >= price > (buy_target - PRICE_CHANGE):
                        return f"→ {row['구간']}"
                except Exception:
                    pass
                return str(row['구간'])  # 항상 문자열로 반환

            grid_df['구간'] = grid_df.apply(add_arrow_to_current_grid, axis=1).astype(str)

            # 표시할 컬럼 선택
            display_columns = ['구간', '매수목표가', '매도목표가', '주문금액', '매수상태', '매수수량', '매수가격', '최종업데이트']
            
            def highlight_current_grid(row):
                try:
                    price = current_price
                    buy_target = float(str(row['매수목표가']).replace('원','').replace(',',''))
                    if buy_target >= price > (buy_target - PRICE_CHANGE):
                        return ['color: red'] * len(row)
                except Exception:
                    pass
                return [''] * len(row)

            styled_grid = grid_df[display_columns].style.apply(highlight_current_grid, axis=1)
            st.write(styled_grid)
        else:
            st.info("현재 활성화된 그리드가 없습니다.")
    
    with trades_container.container():
        # 거래 내역
        st.subheader("거래 내역")
        trades_df = load_trades(7)  # 7일로 고정
        
        if not trades_df.empty:
            # 거래 타입별 색상 설정
            trades_df['color'] = trades_df['type'].map({'buy': 'red', 'sell': 'blue'})
            
            # 거래 내역 테이블
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df['timestamp'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 컬럼명 한글로 변경
            trades_df = trades_df.rename(columns={
                'timestamp': '시간',
                'type': '거래유형',
                'grid_level': '그리드레벨',
                'price': '가격',
                'amount': '거래금액',
                'volume': '거래수량',
                'fee': '수수료',
                'profit': '수익',
                'profit_percentage': '수익률'
            })
            
            # 표시할 컬럼 선택
            display_columns = ['시간', '거래유형', '그리드레벨', '가격', '거래금액', '거래수량', '수수료', '수익', '수익률']
            
            # 데이터 포맷팅
            for col in ['가격', '거래금액', '수수료', '수익']:
                trades_df[col] = trades_df[col].apply(lambda x: f"{x:,.0f}원")
            
            trades_df['수익률'] = trades_df['수익률'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "-")
            trades_df['거래수량'] = trades_df['거래수량'].apply(lambda x: f"{x:.8f}")
            
            st.dataframe(
                trades_df[display_columns],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("조회 기간 내 거래 내역이 없습니다.")

if __name__ == "__main__":
    main()
