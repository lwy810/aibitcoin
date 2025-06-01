import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import time
from streamlit_autorefresh import st_autorefresh  # 자동 새로고침 추가
import streamlit.components.v1 as components  # JavaScript 실행용
import glob  # 파일 패턴 검색용
import os  # 파일 시스템 접근용

# 전역 설정
REFRESH_INTERVAL = 10  # 자동 새로고침 간격 (초)

# 가장 최근 DB 파일 찾기
def find_latest_db_file():
    """현재 폴더에서 가장 최근의 trading_history_*.db 파일을 찾습니다."""
    db_files = glob.glob('trading_history_*.db')
    if not db_files:
        return None
    
    # 파일명으로 정렬 (YYYYMMDDHHMM 형식이므로 파일명 정렬이 시간순 정렬과 같음)
    db_files.sort(reverse=True)  # 최신 파일이 첫 번째로
    return db_files[0]

# 데이터베이스 연결
def get_db_connection():
    latest_db = find_latest_db_file()
    if latest_db is None:
        st.error("trading_history_*.db 파일을 찾을 수 없습니다. 거래 프로그램을 먼저 실행해주세요.")
        st.stop()
    
    return sqlite3.connect(latest_db)

# 스크롤 위치 관리 함수들
def setup_scroll_save():
    """스크롤 이벤트 리스너를 설정하여 스크롤할 때마다 즉시 저장"""
    components.html(
        """
        <script>
        // 스크롤 이벤트 리스너 추가 (중복 방지)
        if (!window.scrollListenerAdded) {
            window.addEventListener('scroll', function() {
                localStorage.setItem('scrollPosition', window.pageYOffset.toString());
            });
            window.scrollListenerAdded = true;
        }
        </script>
        """,
        height=0
    )

def restore_scroll_position():
    """저장된 스크롤 위치로 복원 및 스크롤 이벤트 리스너 설정"""
    components.html(
        """
        <script>
        // 저장된 스크롤 위치로 복원
        window.onload = function() {
            setTimeout(function() {
                const savedPosition = localStorage.getItem('scrollPosition');
                if (savedPosition) {
                    window.scrollTo(0, parseInt(savedPosition));
                }
            }, 100);
        };
        
        // 페이지가 이미 로드된 경우를 위한 즉시 실행
        const savedPosition = localStorage.getItem('scrollPosition');
        if (savedPosition) {
            setTimeout(function() {
                window.scrollTo(0, parseInt(savedPosition));
            }, 100);
        }
        
        // 스크롤 이벤트 리스너 추가 (중복 방지)
        if (!window.scrollListenerAdded) {
            window.addEventListener('scroll', function() {
                localStorage.setItem('scrollPosition', window.pageYOffset.toString());
            });
            window.scrollListenerAdded = true;
        }
        </script>
        """,
        height=0
    )

# 페이지 설정
st.set_page_config(
    page_title="업비트 그리드 트레이딩 대시보드",
    page_icon="📈",
    layout="wide"
)

# TICKER를 데이터베이스에서 가져오는 함수 추가
def get_current_ticker():
    """데이터베이스에서 현재 사용 중인 TICKER를 가져옵니다."""
    conn = get_db_connection()
    try:
        # grid 테이블에서 최근 ticker 조회
        query = "SELECT DISTINCT ticker FROM grid ORDER BY timestamp DESC LIMIT 1"
        result = pd.read_sql_query(query, conn)
        if not result.empty:
            return result['ticker'].iloc[0]
        
        # grid가 비어있으면 trades에서 조회
        query = "SELECT DISTINCT ticker FROM trades ORDER BY timestamp DESC LIMIT 1"
        result = pd.read_sql_query(query, conn)
        if not result.empty:
            return result['ticker'].iloc[0]
        
        # 둘 다 비어있으면 기본값 반환
        return "KRW-XRP"
    except Exception:
        return "KRW-XRP"
    finally:
        conn.close()

# 데이터 로드 함수들
def load_trades(days=7, ticker=None):
    if ticker is None:
        ticker = get_current_ticker()
    
    conn = get_db_connection()
    query = f"""
    SELECT * FROM trades 
    WHERE timestamp >= datetime('now', '-{days} days')
    AND ticker = '{ticker}'
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

def get_summary_stats(ticker=None):
    if ticker is None:
        ticker = get_current_ticker()
        
    conn = get_db_connection()
    
    # 전체 거래 통계 (buy_sell 컬럼 사용)
    trades_query = f"""
    SELECT 
        COALESCE(COUNT(*), 0) as total_trades,
        COALESCE(SUM(CASE WHEN buy_sell = 'buy' THEN 1 ELSE 0 END), 0) as buy_count,
        COALESCE(SUM(CASE WHEN buy_sell = 'sell' THEN 1 ELSE 0 END), 0) as sell_count,
        COALESCE(SUM(CASE WHEN buy_sell = 'buy' THEN amount ELSE 0 END), 0) as total_buy_amount,
        COALESCE(SUM(CASE WHEN buy_sell = 'sell' THEN amount ELSE 0 END), 0) as total_sell_amount,
        COALESCE(SUM(fee), 0) as total_fees,
        COALESCE(SUM(profit), 0) as total_profit
    FROM trades
    WHERE ticker = '{ticker}'
    """
    
    # 최근 잔고 정보
    balance_query = """
    SELECT * FROM balance_history 
    ORDER BY timestamp DESC LIMIT 1
    """
    
    trades_stats = pd.read_sql_query(trades_query, conn)
    latest_balance = pd.read_sql_query(balance_query, conn)
    
    # 빈 결과인 경우 기본값으로 채우기
    if trades_stats.empty:
        trades_stats = pd.DataFrame({
            'total_trades': [0],
            'buy_count': [0],
            'sell_count': [0],
            'total_buy_amount': [0],
            'total_sell_amount': [0],
            'total_fees': [0],
            'total_profit': [0]
        })
    
    conn.close()
    return trades_stats, latest_balance

def load_grid_status(ticker=None):
    """현재 그리드 상태를 가져옵니다."""
    if ticker is None:
        ticker = get_current_ticker()
        
    conn = get_db_connection()
    query = f"""
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
        AND ticker = '{ticker}'
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
    st.title("📈 업비트 그리드 트레이딩 대시보드")
    
    # 스크롤 위치 복원
    restore_scroll_position()
    
    # 동적으로 TICKER 가져오기
    TICKER = get_current_ticker()
    
    # 실제 PRICE_CHANGE 값을 그리드 데이터에서 계산
    grid_df = load_grid_status(TICKER)
    PRICE_CHANGE = 2  # 기본값
    if not grid_df.empty and len(grid_df) >= 2:
        # 연속된 두 그리드의 매수목표가 차이로 PRICE_CHANGE 계산
        price_diff = grid_df['buy_price_target'].iloc[0] - grid_df['buy_price_target'].iloc[1]
        PRICE_CHANGE = abs(price_diff)
    
    # 각 섹션별 컨테이너 생성
    metrics_container = st.empty()
    grid_container = st.empty()
    trades_container = st.empty()
    
    # 초기 데이터 로드 및 표시
    update_dashboard(TICKER, PRICE_CHANGE, grid_df, metrics_container, grid_container, trades_container)
    
    # 자동 업데이트 루프
    while True:
        time.sleep(REFRESH_INTERVAL)
        # 새로운 데이터 로드
        new_grid_df = load_grid_status(TICKER)
        # 데이터 업데이트
        update_dashboard(TICKER, PRICE_CHANGE, new_grid_df, metrics_container, grid_container, trades_container)

def update_dashboard(TICKER, PRICE_CHANGE, grid_df, metrics_container, grid_container, trades_container):
    """대시보드의 각 섹션을 업데이트"""
    
    # 현재가 가져오기
    current_price = None
    try:
        _, latest_balance = get_summary_stats(TICKER)
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
        trades_stats, latest_balance = get_summary_stats(TICKER)
        
        # 상단 메트릭
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_trades = trades_stats['total_trades'].iloc[0] if not trades_stats.empty else 0
            buy_count = trades_stats['buy_count'].iloc[0] if not trades_stats.empty else 0
            sell_count = trades_stats['sell_count'].iloc[0] if not trades_stats.empty else 0
            
            st.metric(
                "총 거래 횟수",
                f"{total_trades:,}회",
                f"매수: {buy_count:,}회 / 매도: {sell_count:,}회"
            )
        
        with col2:
            total_profit = trades_stats['total_profit'].iloc[0] if not trades_stats.empty else 0
            total_fees = trades_stats['total_fees'].iloc[0] if not trades_stats.empty else 0
            profit_color = "normal" if total_profit >= 0 else "inverse"
            st.metric(
                "총 수익",
                f"{total_profit:,.0f}원",
                f"수수료: {total_fees:,.0f}원",
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
                balance_df = load_balance_history(7)  # balance_df 정의 추가
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
        current_time_small = datetime.now().strftime('%H:%M:%S')
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <h3 style="margin: 0; margin-right: 15px;">그리드 현황</h3>
                <span style="
                    font-size: 12px; 
                    color: white;
                    padding: 2px 8px;
                    border-radius: 10px;
                    animation: colorTransition {REFRESH_INTERVAL}s ease-in-out infinite;
                ">
                    🔄 {current_time_small} 업데이트됨
                </span>
            </div>
            <style>
            @keyframes colorTransition {{
                0% {{ 
                    background: linear-gradient(45deg, #606060, #505050);
                }}
                10% {{
                    background: linear-gradient(45deg, #666666, #565656);
                }}
                20% {{
                    background: linear-gradient(45deg, #6c6c6c, #5c5c5c);
                }}
                30% {{
                    background: linear-gradient(45deg, #727272, #626262);
                }}
                40% {{
                    background: linear-gradient(45deg, #787878, #686868);
                }}
                50% {{
                    background: linear-gradient(45deg, #7e7e7e, #6e6e6e);
                }}
                60% {{
                    background: linear-gradient(45deg, #848484, #747474);
                }}
                70% {{
                    background: linear-gradient(45deg, #8a8a8a, #7a7a7a);
                }}
                80% {{
                    background: linear-gradient(45deg, #909090, #808080);
                }}
                90% {{
                    background: linear-gradient(45deg, #969696, #868686);
                }}
                100% {{ 
                    background: linear-gradient(45deg, #9c9c9c, #8c8c8c);
                }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        # grid_df는 이미 위에서 로드했으므로 재사용
        
        if not grid_df.empty:
            # 컬럼명 한글로 변경
            grid_df_display = grid_df.copy()
            grid_df_display = grid_df_display.rename(columns={
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
            grid_df_display['매수목표가'] = grid_df_display['매수목표가'].apply(lambda x: f"{x:,.2f}원")
            grid_df_display['매도목표가'] = grid_df_display['매도목표가'].apply(lambda x: f"{x:,.2f}원")
            grid_df_display['주문금액'] = grid_df_display['주문금액'].apply(lambda x: f"{x:,.0f}원")
            grid_df_display['매수수량'] = grid_df_display['매수수량'].apply(lambda x: f"{x:.8f}" if x > 0 else "-")
            grid_df_display['매수가격'] = grid_df_display['매수가격'].apply(lambda x: f"{x:,.2f}원" if x > 0 else "-")
            grid_df_display['매수상태'] = grid_df_display['매수상태'].apply(lambda x: "매수완료" if x else "대기중")
            grid_df_display['최종업데이트'] = pd.to_datetime(grid_df_display['최종업데이트']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 구간 컬럼에 화살표 추가
            def add_arrow_to_current_grid(row):
                try:
                    price = current_price
                    buy_target = float(str(row['매수목표가']).replace('원','').replace(',',''))
                    sell_target = float(str(row['매도목표가']).replace('원','').replace(',',''))
                    
                    # 현재가가 해당 그리드의 가격 범위에 있는지 확인
                    # 그리드 범위: 매수목표가 < 현재가 <= 매도목표가
                    if buy_target < price <= sell_target:
                        return f"→ {row['구간']}"
                except Exception:
                    pass
                return str(row['구간'])  # 항상 문자열로 반환

            grid_df_display['구간'] = grid_df_display.apply(add_arrow_to_current_grid, axis=1).astype(str)

            # 표시할 컬럼 선택
            display_columns = ['구간', '매수목표가', '매도목표가', '주문금액', '매수상태', '매수수량', '매수가격', '최종업데이트']
            
            def highlight_current_grid(row):
                try:
                    price = current_price
                    buy_target = float(str(row['매수목표가']).replace('원','').replace(',',''))
                    sell_target = float(str(row['매도목표가']).replace('원','').replace(',',''))
                    
                    # 현재가가 해당 그리드의 가격 범위에 있는지 확인
                    # 그리드 범위: 매수목표가 < 현재가 <= 매도목표가
                    if buy_target < price <= sell_target:
                        return ['color: red'] * len(row)
                except Exception:
                    pass
                return [''] * len(row)

            styled_grid = grid_df_display[display_columns].style.apply(highlight_current_grid, axis=1)
            
            # 행 수에 따라 높이 계산 (헤더 + 각 행 * 35픽셀 + 여백)
            table_height = min(len(grid_df_display) * 35 + 50, 800)  # 최대 800픽셀
            
            st.dataframe(
                styled_grid,
                use_container_width=True,
                height=table_height,
                hide_index=True
            )
        else:
            st.info("현재 활성화된 그리드가 없습니다.")
    
    with trades_container.container():
        # 거래 내역
        current_time_small = datetime.now().strftime('%H:%M:%S')
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <h3 style="margin: 0; margin-right: 15px;">거래 내역</h3>
                <span style="
                    font-size: 12px; 
                    color: white;
                    padding: 2px 8px;
                    border-radius: 10px;
                    animation: colorTransition {REFRESH_INTERVAL}s ease-in-out infinite;
                ">
                    📈 {current_time_small} 업데이트됨
                </span>
            </div>
            <style>
            @keyframes colorTransition {{
                0% {{ 
                    background: linear-gradient(45deg, #606060, #505050);
                }}
                10% {{
                    background: linear-gradient(45deg, #666666, #565656);
                }}
                20% {{
                    background: linear-gradient(45deg, #6c6c6c, #5c5c5c);
                }}
                30% {{
                    background: linear-gradient(45deg, #727272, #626262);
                }}
                40% {{
                    background: linear-gradient(45deg, #787878, #686868);
                }}
                50% {{
                    background: linear-gradient(45deg, #7e7e7e, #6e6e6e);
                }}
                60% {{
                    background: linear-gradient(45deg, #848484, #747474);
                }}
                70% {{
                    background: linear-gradient(45deg, #8a8a8a, #7a7a7a);
                }}
                80% {{
                    background: linear-gradient(45deg, #909090, #808080);
                }}
                90% {{
                    background: linear-gradient(45deg, #969696, #868686);
                }}
                100% {{ 
                    background: linear-gradient(45deg, #9c9c9c, #8c8c8c);
                }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        trades_df = load_trades(7, TICKER)  # TICKER 전달
        
        if not trades_df.empty:
            # 거래 타입별 색상 설정 (buy_sell 컬럼 사용)
            trades_df['color'] = trades_df['buy_sell'].map({'buy': 'red', 'sell': 'blue'})
            
            # 거래 내역 테이블
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df['timestamp'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 컬럼명 한글로 변경 (buy_sell -> 거래유형)
            trades_df = trades_df.rename(columns={
                'timestamp': '시간',
                'buy_sell': '거래유형',
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
            
            # 행 수에 따라 높이 계산 (30행까지는 스크롤 없음, 31행부터 스크롤)
            max_rows_without_scroll = 30
            table_height = min(len(trades_df) * 35 + 50, max_rows_without_scroll * 35 + 50)
            
            st.dataframe(
                trades_df[display_columns],
                use_container_width=True,
                height=table_height,
                hide_index=True
            )
        else:
            st.info("조회 기간 내 거래 내역이 없습니다.")

if __name__ == "__main__":
    main()
