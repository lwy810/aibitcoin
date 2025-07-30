import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta, timezone
import time
import streamlit.components.v1 as components
import glob

# nohup python3 bot_coin_grid_cron.py 2>&1 &
# nohup streamlit run bot_dashboard_cron.py --server.port 8502 2>&1 &
# py -m streamlit run bot_dashboard_cron.py --server.port 8502

# 전역 설정
REFRESH_INTERVAL = 60  # 자동 새로고침 간격 (초)

# DB 파일 찾기 (컴팩트 버전)
def find_latest_db_file():
    """최근 trading_history DB 파일을 찾습니다."""
    files = glob.glob('trading_history_*.db')
    return max(files) if files else None

# 데이터베이스 연결
def get_db_connection():
    latest_db = find_latest_db_file()
    if latest_db is None:
        st.error("거래 내역 DB 파일을 찾을 수 없습니다. bot-coin-grid-cron.py를 먼저 실행해주세요.")
        st.stop()
    
    return sqlite3.connect(latest_db)

# 스크롤 위치 관리 (간소화)
def restore_scroll_position():
    """저장된 스크롤 위치로 복원"""
    components.html(
        """
        <script>
        const savedPosition = localStorage.getItem('scrollPosition');
        if (savedPosition) {
            setTimeout(() => window.scrollTo(0, parseInt(savedPosition)), 100);
        }
        
        if (!window.scrollListenerAdded) {
            window.addEventListener('scroll', () => {
                localStorage.setItem('scrollPosition', window.pageYOffset.toString());
            });
            window.scrollListenerAdded = true;
        }
        </script>
        """,
        height=0
    )

# 페이지 설정
def get_grid_start_date():
    """grid 테이블의 가장 오래된 timestamp를 조회합니다."""
    try:
        conn = get_db_connection()
        result = pd.read_sql_query(
            "SELECT MIN(timestamp) as start_date FROM grid", 
            conn
        )
        conn.close()
        
        if not result.empty and pd.notna(result['start_date'].iloc[0]):
            # timestamp를 datetime으로 변환하고 YYYY-MM-DD 형태로 포맷
            start_datetime = pd.to_datetime(result['start_date'].iloc[0])
            return start_datetime.strftime('%Y-%m-%d')
    except Exception:
        pass
    
    # 기본값 반환
    return datetime.now().strftime('%Y-%m-%d')

# 시작일 조회
start_date = get_grid_start_date()

st.set_page_config(
    page_title=f"그리드 트레이딩 대시보드",
    page_icon="📈",
    layout="wide"
)

# TICKER 조회 (컴팩트 버전)
def get_current_ticker():
    """DB에서 현재 TICKER를 가져옵니다."""
    try:
        conn = get_db_connection()
        for table in ['grid', 'trades']:
            result = pd.read_sql_query(f"SELECT DISTINCT ticker FROM {table} ORDER BY timestamp DESC LIMIT 1", conn)
            if not result.empty:
                conn.close()
                return result['ticker'].iloc[0]
        conn.close()
    except Exception:
        pass
    return "KRW-DOGE"

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

def load_balance_history(days=7, ticker=None):
    if ticker is None:
        ticker = get_current_ticker()
    
    conn = get_db_connection()
    query = f"""
    SELECT * FROM balance_history 
    WHERE timestamp >= datetime('now', '-{days} days')
    AND ticker = '{ticker}'
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
    
    # 최근 잔고 정보 (해당 ticker에 대한)
    balance_query = f"""
    SELECT * FROM balance_history 
    WHERE ticker = '{ticker}'
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
        WHERE ticker = '{ticker}'
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
    names = {
        # 업비트 코인 (KRW- 접두사) - 시총 상위 30개
        "KRW-BTC": "비트코인", "KRW-ETH": "이더리움", "KRW-XRP": "리플", "KRW-BNB": "바이낸스코인",
        "KRW-SOL": "솔라나", "KRW-USDT": "테더", "KRW-USDC": "USDC", "KRW-DOGE": "도지코인",
        "KRW-ADA": "에이다", "KRW-TRX": "트론", "KRW-TON": "톤코인", "KRW-AVAX": "아발란체",
        "KRW-SHIB": "시바이누", "KRW-LINK": "체인링크", "KRW-BCH": "비트코인캐시", "KRW-DOT": "폴카닷",
        "KRW-MATIC": "폴리곤", "KRW-LTC": "라이트코인", "KRW-DAI": "다이", "KRW-UNI": "유니스왑",
        "KRW-PEPE": "페페", "KRW-ICP": "인터넷컴퓨터", "KRW-EOS": "이오스", "KRW-XLM": "스텔라루멘",
        "KRW-CRO": "크로노스", "KRW-ATOM": "코스모스", "KRW-HBAR": "헤데라", "KRW-FIL": "파일코인",
        "KRW-VET": "비체인", "KRW-MKR": "메이커", "KRW-AAVE": "아베", "KRW-ALGO": "알고랜드",
        
        # 빗썸 코인 (접두사 없음) - 시총 상위 30개
        "BTC": "비트코인", "ETH": "이더리움", "XRP": "리플", "BNB": "바이낸스코인",
        "SOL": "솔라나", "USDT": "테더", "USDC": "USDC", "DOGE": "도지코인",
        "ADA": "에이다", "TRX": "트론", "TON": "톤코인", "AVAX": "아발란체",
        "SHIB": "시바이누", "LINK": "체인링크", "BCH": "비트코인캐시", "DOT": "폴카닷",
        "MATIC": "폴리곤", "LTC": "라이트코인", "DAI": "다이", "UNI": "유니스왑",
        "PEPE": "페페", "ICP": "인터넷컴퓨터", "EOS": "이오스", "XLM": "스텔라루멘",
        "CRO": "크로노스", "ATOM": "코스모스", "HBAR": "헤데라", "FIL": "파일코인",
        "VET": "비체인", "MKR": "메이커", "AAVE": "아베", "ALGO": "알고랜드",
        "SEI": "세이"
    }
    return names.get(ticker, ticker)

def get_latest_price(ticker=None):
    if ticker is None:
        ticker = get_current_ticker()
    
    conn = get_db_connection()
    df = pd.read_sql_query(f"SELECT current_price FROM balance_history WHERE ticker = '{ticker}' ORDER BY timestamp DESC LIMIT 1", conn)
    conn.close()
    return df['current_price'].iloc[0] if not df.empty else None

# 메인 대시보드
def main():
    global start_date  # 전역 변수 선언
    st.markdown(f"### 📈 그리드 트레이딩 대시보드(시작일: {start_date})")
    
    # 스크롤 위치 복원
    restore_scroll_position()
    
    # 동적으로 TICKER 가져오기
    TICKER = get_current_ticker()
    
    # PRICE_CHANGE 계산
    grid_df = load_grid_status(TICKER)
    PRICE_CHANGE = abs(grid_df['buy_price_target'].iloc[0] - grid_df['buy_price_target'].iloc[1]) if len(grid_df) >= 2 else 2
    
    # 컨테이너 생성
    metrics_container = st.empty()
    grid_container = st.empty()
    trades_container = st.empty()
    
    # 대시보드 업데이트 루프
    while True:
        grid_df = load_grid_status(TICKER)
        update_dashboard(TICKER, PRICE_CHANGE, grid_df, metrics_container, grid_container, trades_container, 7, True)
        time.sleep(REFRESH_INTERVAL)

def update_dashboard(TICKER, PRICE_CHANGE, grid_df, metrics_container, grid_container, trades_container, data_days=7, show_all_grids=True):
    """대시보드의 각 섹션을 업데이트"""
    
    # 현재가 조회
    try:
        _, latest_balance = get_summary_stats(TICKER)
        current_price = latest_balance['current_price'].iloc[0] if not latest_balance.empty else None
    except Exception:
        current_price = None

    # 코인명 먼저 정의
    coin_name = get_coin_name(TICKER)
    
    with metrics_container.container():
        # 코인명/현재가 출력 (메트릭 위로 이동)
        if current_price is not None:
            st.markdown(f"#### {TICKER} ({coin_name}) | 현재가: **{current_price:,.2f}원**")
        else:
            st.markdown(f"#### {TICKER} ({coin_name}) | 현재가: -")
        # 요약 통계 (설정된 기간)
        trades_stats, latest_balance = get_summary_stats(TICKER)
        
        # 상단 메트릭
        col1, col2, col3 = st.columns(3)
        
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
            
            # 그리드 설정액 대비 수익률 계산
            profit_percentage_on_grid_text = "" 
            if not grid_df.empty and 'order_krw_amount' in grid_df.columns:
                # order_krw_amount는 그리드 전체에 동일하다고 가정하고 첫 번째 값을 사용
                order_krw_amount_value = grid_df['order_krw_amount'].iloc[0] 
                num_grid_levels = len(grid_df)

                # order_krw_amount_value가 유효한 숫자인지, 0보다 큰지, 그리드 레벨이 있는지 확인
                if pd.notna(order_krw_amount_value) and order_krw_amount_value > 0 and num_grid_levels > 0:
                    total_potential_investment = num_grid_levels * order_krw_amount_value
                    # total_potential_investment가 0이 아닐 때만 계산 (0으로 나누기 방지)
                    profit_vs_potential_investment_pct = (total_profit / total_potential_investment) * 100 if total_potential_investment != 0 else 0
                    profit_percentage_on_grid_text = f"수익률: {profit_vs_potential_investment_pct:+.2f}%"
                else:
                    profit_percentage_on_grid_text = "그리드 정보 계산 불가" # 계산에 필요한 정보 부족
            else:
                profit_percentage_on_grid_text = "그리드 정보 없음" # grid_df가 비어있거나 필요한 컬럼 부재

            current_delta_text = f"수수료: {total_fees:,.0f}원"
            if profit_percentage_on_grid_text: # 계산된 수익률 정보가 있으면 추가
                # 연평균 수익률 계산
                try:
                    balance_df = load_balance_history(365, TICKER)
                    if not balance_df.empty and len(balance_df) > 1:
                        # 가장 오래된 기록의 날짜
                        start_date = pd.to_datetime(balance_df['timestamp'].iloc[0])
                        current_date = pd.to_datetime(datetime.now())
                        days_elapsed = (current_date - start_date).days
                        
                        if days_elapsed > 0 and total_potential_investment > 0:
                            # 연평균 수익률 계산: (1 + 총수익률) ^ (365 / 경과일수) - 1
                            total_return_ratio = total_profit / total_potential_investment
                            annual_return = ((1 + total_return_ratio) ** (365 / days_elapsed) - 1) * 100
                            annual_return_text = f"연수익률: {annual_return:+.1f}%"
                            current_delta_text += f" | {profit_percentage_on_grid_text} | {annual_return_text}"
                        else:
                            current_delta_text += f" | {profit_percentage_on_grid_text}"
                    else:
                        current_delta_text += f" | {profit_percentage_on_grid_text}"
                except Exception:
                    current_delta_text += f" | {profit_percentage_on_grid_text}"
            
            st.metric(
                "순수익(총수익-수수료)",
                f"{total_profit:,.0f}원",
                delta=current_delta_text,
                delta_color=profit_color
            )
        
        with col3:
            if not latest_balance.empty:
                coin_value = latest_balance['coin_balance'].iloc[0] * latest_balance['current_price'].iloc[0]
                coin_balance = latest_balance['coin_balance'].iloc[0]
                coin_avg_price = latest_balance['coin_avg_price'].iloc[0]
                
                # 현재 보유 코인의 평균 매수 가치 계산
                if coin_balance > 0 and coin_avg_price > 0:
                    avg_buy_value = coin_balance * coin_avg_price
                    ratio = coin_value / avg_buy_value
                    ratio_pct = (ratio - 1) * 100  # 수익률 계산
                    display_text = f"{coin_value:,.0f}원 / {avg_buy_value:,.0f}원"
                    delta_text = f"비율: {ratio:.3f} ({ratio_pct:+.2f}%)"
                else:
                    display_text = f"{coin_value:,.0f}원"
                    delta_text = "평균 매수가 데이터 없음"

                st.metric(
                    f"{coin_name} 현재가치/평균매수가치",
                    display_text,
                    delta_text
                )
        
    
    with grid_container.container():
        # 그리드 현황
        kst = timezone(timedelta(hours=9))
        current_time_kst = datetime.now(kst)
        current_time_small = current_time_kst.strftime('%H:%M:%S')
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <h4 style="margin: 0; margin-right: 15px;">그리드 현황</h4>
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
            # 그리드 필터링 (show_all_grids 옵션 적용)
            if not show_all_grids:
                grid_df_filtered = grid_df[grid_df['is_bought'] == True]
            else:
                grid_df_filtered = grid_df
            
            # 필터링 후 데이터가 있는지 확인
            if grid_df_filtered.empty:
                st.info("매수된 그리드가 없습니다.")
            else:
                # 컬럼명 한글로 변경
                grid_df_display = grid_df_filtered.copy()
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
                
                # 그리드 현황은 모든 행을 표시하도록 height 파라미터 제거
                st.dataframe(
                    grid_df_display[display_columns], # 스타일이 적용되지 않은 DataFrame을 직접 전달
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("현재 활성화된 그리드가 없습니다.")
    
    with trades_container.container():
        # 거래 내역
        kst = timezone(timedelta(hours=9))
        current_time_kst = datetime.now(kst)
        current_time_small = current_time_kst.strftime('%H:%M:%S')
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <h4 style="margin: 0; margin-right: 15px;">거래 내역</h4>
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
            
            st.dataframe(
                trades_df[display_columns],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("조회 기간 내 거래 내역이 없습니다.")

if __name__ == "__main__":
    main()