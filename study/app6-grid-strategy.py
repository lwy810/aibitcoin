import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import pyupbit
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="그리드 현물매매 전략 백테스트", layout="wide")

st.title("📊 그리드 현물매매 전략 백테스트")

# 사이드바 설정
st.sidebar.header("백테스트 설정")

# 기본 설정
ticker = st.sidebar.selectbox("거래 코인", ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-DOGE"], index=2)
initial_balance = st.sidebar.number_input("초기 자금 (원)", value=1000000, step=100000)
test_days = st.sidebar.selectbox("백테스트 기간", [10, 30, 60, 90, 180, 365], index=1)

# 그리드 전략 설정
st.sidebar.subheader("그리드 전략 설정")
price_change = st.sidebar.number_input("그리드 간격 (원)", value=4, min_value=1, step=1, 
                                     help="각 그리드 레벨 간의 가격 차이")
offset_grid = st.sidebar.slider("기준가 오프셋 (구간)", 0, 20, 4, 1,
                               help="현재가로부터 몇 구간 위를 기준가로 설정할지")

st.sidebar.markdown("**기준가격은 현재가로 설정됩니다.**")
# 기준가격 직접 입력 옵션 추가
use_custom_base_price = st.sidebar.checkbox("기준가격 직접 설정", value=False,
                                          help="체크하면 기준가격을 직접 입력할 수 있습니다.")
custom_base_price = None
if use_custom_base_price:
    custom_base_price = st.sidebar.number_input("기준가격 (원)", value=100, min_value=1, step=1,
                                               help="그리드의 중심이 될 기준가격을 직접 설정")

order_amount = st.sidebar.number_input("주문당 금액 (원)", value=50000, min_value=5000, step=5000,
                                     help="각 그리드 레벨에서의 주문 금액")
max_grid_count = st.sidebar.slider("최대 그리드 수", 5, 50, 10, 1,
                                 help="설정할 그리드 레벨의 최대 개수")
fee_rate = st.sidebar.slider("거래 수수료 (%)", 0.0, 1.0, 0.05, 0.01,
                           help="매수/매도 시 적용될 수수료율") / 100

# 백테스트 실행 버튼
if st.sidebar.button("백테스트 실행"):
    with st.spinner("백테스트 실행 중..."):
        
        @st.cache_data
        def get_price_data(ticker, days):
            """가격 데이터 조회"""
            df = pyupbit.get_ohlcv(ticker, interval="minute60", count=days*24)  # 1시간 봉 데이터
            if df is not None and not df.empty:
                df = df.reset_index()
                df = df.rename(columns={'index': 'datetime'})
                return df
            return None
        
        # 데이터 수집
        df = get_price_data(ticker, test_days)
        
        if df is not None and len(df) > 1:
            # 백테스트 변수 초기화
            krw_balance = initial_balance
            coin_balance = 0
            grid_orders = []
            trade_history = []
            balance_history = []
            
            # 기준가 설정 (사용자 입력 우선, 없으면 자동 계산)
            if use_custom_base_price and custom_base_price:
                base_price = custom_base_price
            else:
                base_price = df.iloc[0]['close'] + (price_change * offset_grid)
            
            # 그리드 주문 생성
            for i in range(max_grid_count):
                buy_target_price = base_price - (i * price_change)
                sell_target_price = buy_target_price + price_change
                
                grid = {
                    'level': i + 1,
                    'buy_price_target': buy_target_price,
                    'sell_price_target': sell_target_price,
                    'order_krw_amount': order_amount,
                    'is_bought': False,
                    'actual_bought_volume': 0.0,
                    'actual_buy_fill_price': 0.0
                }
                grid_orders.append(grid)
            
            # 백테스트 시뮬레이션
            for idx, row in df.iterrows():
                current_price = row['close']
                current_time = row['datetime']
                
                # 각 그리드 레벨에 대해 매수/매도 조건 확인
                for grid in grid_orders:
                    # 매수 조건: 현재 가격이 매수 목표가 이하이고 아직 매수되지 않은 경우
                    if not grid['is_bought'] and current_price <= grid['buy_price_target']:
                        if krw_balance >= grid['order_krw_amount']:
                            # 매수 실행
                            fee_paid = grid['order_krw_amount'] * fee_rate
                            krw_for_coin = grid['order_krw_amount'] * (1 - fee_rate)
                            bought_volume = krw_for_coin / current_price
                            
                            grid['is_bought'] = True
                            grid['actual_bought_volume'] = bought_volume
                            grid['actual_buy_fill_price'] = current_price
                            
                            krw_balance -= grid['order_krw_amount']
                            coin_balance += bought_volume
                            
                            # 거래 기록
                            trade_history.append({
                                'datetime': current_time,
                                'type': 'BUY',
                                'level': grid['level'],
                                'price': current_price,
                                'volume': bought_volume,
                                'amount': grid['order_krw_amount'],
                                'fee': fee_paid
                            })
                    
                    # 매도 조건: 현재 가격이 매도 목표가 이상이고 이미 매수된 경우
                    elif grid['is_bought'] and current_price >= grid['sell_price_target']:
                        # 매도 실행
                        sell_volume = grid['actual_bought_volume']
                        gross_sell_amount = sell_volume * current_price
                        fee_paid = gross_sell_amount * fee_rate
                        net_sell_amount = gross_sell_amount * (1 - fee_rate)
                        
                        grid['is_bought'] = False
                        coin_balance -= sell_volume
                        krw_balance += net_sell_amount
                        
                        # 거래 기록
                        trade_history.append({
                            'datetime': current_time,
                            'type': 'SELL',
                            'level': grid['level'],
                            'price': current_price,
                            'volume': sell_volume,
                            'amount': net_sell_amount,
                            'fee': fee_paid
                        })
                        
                        # 그리드 상태 초기화
                        grid['actual_bought_volume'] = 0.0
                        grid['actual_buy_fill_price'] = 0.0
                
                # 현재 자산 가치 기록
                current_coin_value = coin_balance * current_price
                total_value = krw_balance + current_coin_value
                
                balance_history.append({
                    'datetime': current_time,
                    'krw_balance': krw_balance,
                    'coin_balance': coin_balance,
                    'coin_value': current_coin_value,
                    'total_value': total_value,
                    'price': current_price
                })
            
            # 최종 결과 계산
            final_balance = balance_history[-1]['total_value'] if balance_history else initial_balance
            total_return = ((final_balance - initial_balance) / initial_balance) * 100
            
            # 매수후보유 수익률
            buy_hold_return = ((df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close']) * 100
            
            # MDD 계산
            balance_series = pd.Series([item['total_value'] for item in balance_history])
            cumulative_max = balance_series.cummax()
            drawdown = (balance_series - cumulative_max) / cumulative_max
            mdd = drawdown.min() * 100
            
            # 거래 통계
            buy_trades = [t for t in trade_history if t['type'] == 'BUY']
            sell_trades = [t for t in trade_history if t['type'] == 'SELL']
            
            # 결과 표시
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("총 수익률", f"{total_return:.2f}%", 
                         delta=f"{total_return:.2f}%")
            
            with col2:
                st.metric("매수후보유 수익률", f"{buy_hold_return:.2f}%",
                         delta=f"{buy_hold_return:.2f}%")
            
            with col3:
                st.metric("최대 낙폭 (MDD)", f"{mdd:.2f}%",
                         delta=f"{mdd:.2f}%" if mdd < 0 else None)
            
            with col4:
                st.metric("총 거래횟수", f"{len(trade_history)}회")
            
            with col5:
                st.metric("최종 자산", f"{final_balance:,.0f}원",
                         delta=f"{final_balance - initial_balance:,.0f}원")
            
            # 차트 생성
            balance_df = pd.DataFrame(balance_history)
            
            # 포트폴리오 가치 변화 차트
            fig1 = go.Figure()
            
            # 포트폴리오 가치
            fig1.add_trace(go.Scatter(
                x=balance_df['datetime'],
                y=balance_df['total_value'],
                mode='lines',
                name='포트폴리오 가치',
                line=dict(color='#00BFFF', width=2)
            ))
            
            # 매수후보유 벤치마크
            benchmark_values = []
            for _, row in balance_df.iterrows():
                benchmark_value = initial_balance * (row['price'] / df.iloc[0]['close'])
                benchmark_values.append(benchmark_value)
            
            fig1.add_trace(go.Scatter(
                x=balance_df['datetime'],
                y=benchmark_values,
                mode='lines',
                name='매수후보유',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            fig1.update_layout(
                title="포트폴리오 가치 변화",
                xaxis_title="시간",
                yaxis_title="자산 가치 (원)",
                plot_bgcolor="#18181c",
                paper_bgcolor="#18181c",
                font=dict(color="white"),
                legend=dict(x=0, y=1)
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # 가격 차트 및 그리드 레벨 표시
            fig2 = go.Figure()
            
            # 가격 차트 (캔들스틱)
            fig2.add_trace(go.Candlestick(
                x=df['datetime'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=f'{ticker} 가격'
            ))
            
            # 그리드 레벨 라인들
            for grid in grid_orders:
                fig2.add_hline(
                    y=grid['buy_price_target'],
                    line_dash="dot",
                    line_color="green",
                    opacity=0.5,
                    annotation_text=f"L{grid['level']} 매수"
                )
                fig2.add_hline(
                    y=grid['sell_price_target'],
                    line_dash="dot", 
                    line_color="red",
                    opacity=0.5,
                    annotation_text=f"L{grid['level']} 매도"
                )
            
            # 기준가 라인
            fig2.add_hline(
                y=base_price,
                line_dash="solid",
                line_color="yellow",
                line_width=2,
                annotation_text="기준가"
            )
            
            # 매수/매도 신호 표시
            if buy_trades:
                buy_times = [t['datetime'] for t in buy_trades]
                buy_prices = [t['price'] for t in buy_trades]
                fig2.add_trace(go.Scatter(
                    x=buy_times,
                    y=buy_prices,
                    mode='markers',
                    name='매수',
                    marker=dict(color='blue', size=8, symbol='triangle-up')
                ))
            
            if sell_trades:
                sell_times = [t['datetime'] for t in sell_trades]
                sell_prices = [t['price'] for t in sell_trades]
                fig2.add_trace(go.Scatter(
                    x=sell_times,
                    y=sell_prices,
                    mode='markers',
                    name='매도',
                    marker=dict(color='red', size=8, symbol='triangle-down')
                ))
            
            fig2.update_layout(
                title=f"{ticker} 가격 및 그리드 전략",
                xaxis_title="시간",
                yaxis_title="가격 (원)",
                plot_bgcolor="#18181c",
                paper_bgcolor="#18181c",
                font=dict(color="white"),
                legend=dict(x=0, y=1),
                xaxis_rangeslider_visible=False,
                xaxis_rangeselector=dict(visible=False)
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # 그리드 상태 표시
            st.subheader("현재 그리드 상태")
            grid_status = []
            for grid in grid_orders:
                status = "매수완료" if grid['is_bought'] else "대기중"
                grid_status.append({
                    '레벨': grid['level'],
                    '매수가': f"{grid['buy_price_target']:,.0f}원",
                    '매도가': f"{grid['sell_price_target']:,.0f}원",
                    '상태': status,
                    '보유량': f"{grid['actual_bought_volume']:.8f}" if grid['is_bought'] else "-"
                })
            
            grid_df = pd.DataFrame(grid_status)
            st.dataframe(grid_df, use_container_width=True, hide_index=True)
            
            # 거래 내역
            if trade_history:
                st.subheader("거래 내역")
                trades_df = pd.DataFrame(trade_history)
                trades_df['datetime'] = trades_df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
                trades_df['price'] = trades_df['price'].apply(lambda x: f"{x:,.0f}원")
                trades_df['volume'] = trades_df['volume'].apply(lambda x: f"{x:.8f}")
                trades_df['amount'] = trades_df['amount'].apply(lambda x: f"{x:,.0f}원")
                trades_df['fee'] = trades_df['fee'].apply(lambda x: f"{x:,.0f}원")
                
                trades_df.columns = ['시간', '거래타입', '레벨', '가격', '수량', '금액', '수수료']
                st.dataframe(trades_df, use_container_width=True, hide_index=True)
            
            # 전략 설명
            st.subheader("📋 그리드 현물매매 전략이란?")
            
            # 기준가 설정 방식 설명 문구 생성
            if use_custom_base_price and custom_base_price:
                base_price_desc = f"사용자 직접 설정 = {base_price:,.0f}원"
            else:
                base_price_desc = f"현재가 + (그리드간격 × 오프셋) = {base_price:,.0f}원"
            
            st.markdown(f"""
            **그리드 현물매매 전략**은 다음과 같은 원리로 작동합니다:
            
            1. **기준가 설정**: {base_price_desc}
            2. **그리드 레벨 생성**: 기준가를 중심으로 {price_change}원 간격으로 {max_grid_count}개 레벨 설정
            3. **매수 조건**: 가격이 각 레벨의 매수가 이하로 떨어지면 {order_amount:,}원 매수
            4. **매도 조건**: 매수한 레벨에서 가격이 매도가(매수가+{price_change}원) 이상으로 오르면 매도
            5. **수수료**: 매수/매도 시 각각 {fee_rate*100:.2f}% 적용
            
            이 전략은 **횡보장에서 효과적**이며, 가격 변동을 통해 지속적인 수익을 추구합니다.
            """)
            
        else:
            st.error("데이터를 불러올 수 없습니다. 네트워크 연결을 확인해주세요.")

else:
    st.info("👈 사이드바에서 설정을 조정하고 '백테스트 실행' 버튼을 클릭하세요.")
    
    # 그리드 전략 설명
    st.markdown("""
    ## 🎯 그리드 현물매매 전략이란?
    
    그리드 전략은 **정해진 가격 구간에서 분할 매수/매도**를 반복하는 전략입니다.
    
    ### 📈 전략 원리
    - 기준가를 중심으로 일정 간격의 그리드 레벨 설정
    - 각 레벨에서 가격 하락 시 매수, 상승 시 매도
    - 횡보장에서 가격 변동을 이용한 수익 창출
    
    ### ⚙️ 주요 파라미터
    - **그리드 간격**: 각 레벨 간의 가격 차이 (원)
    - **기준가 오프셋**: 현재가로부터 몇 구간 위를 기준가로 설정
    - **주문당 금액**: 각 그리드 레벨에서의 투자 금액
    - **최대 그리드 수**: 설정할 그리드 레벨의 개수
    
    ### 📊 백테스트 결과
    - 총 수익률과 매수후보유 수익률 비교
    - 그리드 레벨별 매수/매도 현황
    - 거래 빈도 및 개별 거래 내역
    - 포트폴리오 가치 변화 추이
    """)
