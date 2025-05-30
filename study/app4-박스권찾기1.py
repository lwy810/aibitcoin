import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import pyupbit
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(page_title="리플 1개월 라인차트", layout="wide")

# 1개월치 데이터 가져오기 (일봉)
df = pyupbit.get_ohlcv("KRW-XRP", interval="day", count=30)

if df is not None and not df.empty:
    df = df.reset_index()
    df = df.rename(columns={"index": "날짜", "close": "종가"})
    
    # 최고가/최저가 정보
    최고가 = df["종가"].max()
    최저가 = df["종가"].min()
    최고가_날짜 = df.loc[df["종가"].idxmax(), "날짜"]
    최저가_날짜 = df.loc[df["종가"].idxmin(), "날짜"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.array(df["날짜"]),
        y=df["종가"],
        mode="lines",
        line=dict(color="#00BFFF", width=3),  # 리플 컬러(파랑)
        name="리플"
    ))
    # 최고가 마커
    fig.add_trace(go.Scatter(
        x=[np.array([최고가_날짜])[0]],
        y=[최고가],
        mode="markers+text",
        marker=dict(color="#00BFFF", size=10),
        text=[f"최고 {최고가:,.0f}"],
        textposition="top right",
        showlegend=False
    ))
    # 최저가 마커
    fig.add_trace(go.Scatter(
        x=[np.array([최저가_날짜])[0]],
        y=[최저가],
        mode="markers+text",
        marker=dict(color="orange", size=10),
        text=[f"최저 {최저가:,.0f}"],
        textposition="bottom left",
        showlegend=False
    ))
    # 빨간색 박스 추가 (5월 10일~15일, 3500~3300원)
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=pd.Timestamp("2025-05-10"),
        x1=pd.Timestamp("2025-05-20"),
        y0=3100,
        y1=3500,
        fillcolor="rgba(255,0,0,0.2)",
        line=dict(color="red", width=2),
        layer="below"
    )
    fig.update_layout(
        title="리플 1개월 가격 추이 (업비트)",
        xaxis_title="날짜",
        yaxis_title="가격(원)",
        plot_bgcolor="#18181c",
        paper_bgcolor="#18181c",
        font=dict(color="white"),
        showlegend=False
    )
    fig.update_xaxes(
        tickformat="%y-%m-%d"  # 년-월-일 형식 (예: 25-05-02)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("업비트에서 데이터를 불러올 수 없습니다. pyupbit 설치 및 네트워크 상태를 확인하세요.")
