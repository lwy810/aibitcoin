import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import pyupbit
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="비트코인 변동성 돌파 전략 백테스트", layout="wide")

st.title("📈 비트코인 변동성 돌파 전략 백테스트")

# 사이드바 설정
st.sidebar.header("백테스트 설정")
k_value = st.sidebar.slider("K값 (변동성 돌파 비율)", 0.1, 1.0, 0.5, 0.1)
initial_balance = st.sidebar.number_input("초기 자금 (원)", value=1000000, step=100000)
test_days = st.sidebar.selectbox("백테스트 기간", [30, 60, 90, 180, 365], index=2)

# 백테스트 실행 버튼
if st.sidebar.button("백테스트 실행"):
    with st.spinner("백테스트 실행 중..."):
        # 데이터 수집
        @st.cache_data
        def get_bitcoin_data(days):
            df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=days+1)
            if df is not None and not df.empty:
                df = df.reset_index()
                # pyupbit 실제 컬럼: ['index', 'open', 'high', 'low', 'close', 'volume', 'value']
                df = df.rename(columns={'index': 'date'})
                # 필요한 컬럼만 선택
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                return df
            return None
        
        df = get_bitcoin_data(test_days)
        
        if df is not None and len(df) > 1:
            # 변동성 돌파 전략 백테스트
            df['prev_high'] = df['high'].shift(1)
            df['prev_low'] = df['low'].shift(1)
            df['volatility'] = df['prev_high'] - df['prev_low']
            df['target_price'] = df['open'] + (df['volatility'] * k_value)
            
            # 거래 시뮬레이션
            balance = initial_balance
            bitcoin_amount = 0
            trades = []
            daily_balance = []
            
            for i in range(1, len(df)):
                current_day = df.iloc[i]
                prev_day = df.iloc[i-1]
                
                # 매수 조건: 현재가가 목표가를 돌파
                if current_day['high'] >= current_day['target_price'] and bitcoin_amount == 0:
                    # 목표가에서 매수
                    buy_price = current_day['target_price']
                    bitcoin_amount = balance / buy_price
                    balance = 0
                    
                    trades.append({
                        'date': current_day['date'],
                        'type': 'BUY',
                        'price': buy_price,
                        'amount': bitcoin_amount,
                        'balance': balance
                    })
                
                # 매도 조건: 다음날 시가에 매도 (하루 보유)
                elif bitcoin_amount > 0:
                    sell_price = current_day['open']
                    balance = bitcoin_amount * sell_price
                    bitcoin_amount = 0
                    
                    trades.append({
                        'date': current_day['date'],
                        'type': 'SELL',
                        'price': sell_price,
                        'amount': 0,
                        'balance': balance
                    })
                
                # 일일 잔고 기록
                current_value = balance + (bitcoin_amount * current_day['close'])
                daily_balance.append({
                    'date': current_day['date'],
                    'balance': current_value,
                    'bitcoin_price': current_day['close']
                })
            
            # 결과 계산
            final_balance = daily_balance[-1]['balance'] if daily_balance else initial_balance
            total_return = ((final_balance - initial_balance) / initial_balance) * 100
            
            # 벤치마크 (매수 후 보유)
            buy_hold_return = ((df.iloc[-1]['close'] - df.iloc[1]['open']) / df.iloc[1]['open']) * 100
            
            # 결과 표시
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("총 수익률", f"{total_return:.2f}%", delta=f"{total_return:.2f}%")
            
            with col2:
                st.metric("매수후보유 수익률", f"{buy_hold_return:.2f}%", delta=f"{buy_hold_return:.2f}%")
            
            with col3:
                st.metric("총 거래횟수", f"{len(trades)}회")
            
            with col4:
                st.metric("최종 자산", f"{final_balance:,.0f}원", delta=f"{final_balance - initial_balance:,.0f}원")
            
            # 차트 생성
            balance_df = pd.DataFrame(daily_balance)
            price_df = df[['date', 'close', 'target_price']].copy()
            
            # 잔고 변화 차트
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=balance_df['date'],
                y=balance_df['balance'],
                mode='lines',
                name='포트폴리오 가치',
                line=dict(color='#00BFFF', width=2)
            ))
            
            # 벤치마크 라인 추가
            benchmark_values = []
            for i, row in balance_df.iterrows():
                benchmark_value = initial_balance * (row['bitcoin_price'] / df.iloc[1]['open'])
                benchmark_values.append(benchmark_value)
            
            fig1.add_trace(go.Scatter(
                x=balance_df['date'],
                y=benchmark_values,
                mode='lines',
                name='매수후보유',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            fig1.update_layout(
                title="포트폴리오 가치 변화",
                xaxis_title="날짜",
                yaxis_title="자산 가치 (원)",
                plot_bgcolor="#18181c",
                paper_bgcolor="#18181c",
                font=dict(color="white"),
                legend=dict(x=0, y=1)
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # 비트코인 가격 및 매매 신호 차트
            fig2 = go.Figure()
            
            # 비트코인 가격
            fig2.add_trace(go.Scatter(
                x=price_df['date'],
                y=price_df['close'],
                mode='lines',
                name='비트코인 가격',
                line=dict(color='#FFD700', width=2)
            ))
            
            # 목표가 라인
            fig2.add_trace(go.Scatter(
                x=price_df['date'],
                y=price_df['target_price'],
                mode='lines',
                name=f'목표가 (K={k_value})',
                line=dict(color='red', width=1, dash='dot')
            ))
            
            # 매수/매도 신호 표시
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            
            if buy_trades:
                fig2.add_trace(go.Scatter(
                    x=[t['date'] for t in buy_trades],
                    y=[t['price'] for t in buy_trades],
                    mode='markers',
                    name='매수',
                    marker=dict(color='blue', size=8, symbol='triangle-up')
                ))
            
            if sell_trades:
                fig2.add_trace(go.Scatter(
                    x=[t['date'] for t in sell_trades],
                    y=[t['price'] for t in sell_trades],
                    mode='markers',
                    name='매도',
                    marker=dict(color='red', size=8, symbol='triangle-down')
                ))
            
            fig2.update_layout(
                title="비트코인 가격 및 매매 신호",
                xaxis_title="날짜",
                yaxis_title="가격 (원)",
                plot_bgcolor="#18181c",
                paper_bgcolor="#18181c",
                font=dict(color="white"),
                legend=dict(x=0, y=1)
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # 거래 내역 테이블
            if trades:
                st.subheader("거래 내역")
                trades_df = pd.DataFrame(trades)
                trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
                trades_df['price'] = trades_df['price'].apply(lambda x: f"{x:,.0f}원")
                trades_df['amount'] = trades_df['amount'].apply(lambda x: f"{x:.8f} BTC" if x > 0 else "-")
                trades_df['balance'] = trades_df['balance'].apply(lambda x: f"{x:,.0f}원")
                
                trades_df.columns = ['날짜', '거래타입', '가격', '수량', '잔고']
                st.dataframe(trades_df, use_container_width=True, hide_index=True)
            
            # 전략 설명
            st.subheader("📋 변동성 돌파 전략이란?")
            st.markdown("""
            **변동성 돌파 전략**은 다음과 같은 원리로 작동합니다:
            
            1. **목표가 계산**: 당일 시가 + (전일 고가-저가) × K값
            2. **매수 조건**: 당일 고가가 목표가를 돌파할 때 목표가에서 매수
            3. **매도 조건**: 다음날 시가에서 매도 (1일 보유)
            4. **K값**: 0.5가 일반적이며, 높을수록 보수적, 낮을수록 공격적
            
            이 전략은 강한 상승 모멘텀을 포착하려는 추세추종 전략입니다.
            """)
            
        else:
            st.error("데이터를 불러올 수 없습니다. 네트워크 연결을 확인해주세요.")

else:
    st.info("👈 사이드바에서 설정을 조정하고 '백테스트 실행' 버튼을 클릭하세요.")
    
    # 변동성 돌파 전략 설명
    st.markdown("""
    ## 🎯 변동성 돌파 전략이란?
    
    변동성 돌파 전략은 **래리 윌리엄스**가 개발한 단기 추세추종 전략입니다.
    
    ### 📈 전략 원리
    - 전일 변동성(고가-저가)을 이용해 당일 목표가 설정
    - 목표가 = 당일시가 + (전일변동성 × K값)
    - 가격이 목표가를 돌파하면 매수 신호
    
    ### ⚙️ 주요 파라미터
    - **K값**: 0.1~1.0 (일반적으로 0.5 사용)
    - **보유기간**: 1일 (당일 매수 → 익일 매도)
    
    ### 📊 백테스트 결과
    - 총 수익률과 매수후보유 수익률 비교
    - 거래 빈도 및 개별 거래 내역
    - 포트폴리오 가치 변화 추이
    """)
