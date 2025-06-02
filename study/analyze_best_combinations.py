import json
import pandas as pd
import numpy as np
from datetime import datetime
import glob

def calculate_risk_adjusted_score(return_pct, mdd_pct, win_rate, num_trades):
    """
    리스크 조정 점수 계산
    - 수익률이 높을수록 좋음
    - MDD가 낮을수록 좋음 (절댓값)
    - 승률이 높을수록 좋음
    - 적절한 거래 횟수 (너무 적거나 많으면 감점)
    """
    
    # 기본 점수: 수익률
    base_score = return_pct
    
    # MDD 페널티 (MDD가 클수록 감점)
    mdd_penalty = abs(mdd_pct) * 0.5  # MDD의 50%를 페널티로 적용
    
    # 승률 보너스
    win_rate_bonus = win_rate * 0.1  # 승률의 10%를 보너스로 적용
    
    # 거래 횟수 조정 (적절한 거래 횟수는 5-50회)
    if num_trades == 0:
        trade_penalty = -50  # 거래 없음 큰 페널티
    elif num_trades < 3:
        trade_penalty = -20  # 너무 적은 거래
    elif num_trades > 100:
        trade_penalty = -10  # 너무 많은 거래 (과최적화)
    else:
        trade_penalty = 0
    
    # 최종 점수
    final_score = base_score - mdd_penalty + win_rate_bonus + trade_penalty
    
    return final_score

def calculate_sharpe_like_ratio(return_pct, mdd_pct):
    """샤프 비율과 유사한 리스크 조정 수익률"""
    if abs(mdd_pct) < 0.1:  # MDD가 거의 0인 경우
        return return_pct * 10  # 높은 점수 부여
    return return_pct / abs(mdd_pct)

def analyze_results():
    # 가장 최근 결과 파일 찾기
    result_files = glob.glob('hyperparameter_tuning_results_*.json')
    if not result_files:
        print("결과 파일을 찾을 수 없습니다.")
        return
    
    latest_file = max(result_files)
    print(f"분석할 파일: {latest_file}")
    
    # 결과 로딩
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"총 {len(results)}개 결과 분석 중...")
    
    # 유효한 결과만 필터링 (수익률이 있는 것들)
    valid_results = []
    for result in results:
        br = result['backtest_results']
        if br['total_return_percent'] != 0 or br['num_trades'] > 0:
            valid_results.append(result)
    
    print(f"유효한 결과: {len(valid_results)}개")
    
    # 각 결과에 대해 점수 계산
    for result in valid_results:
        br = result['backtest_results']
        
        # 리스크 조정 점수 계산
        risk_adjusted_score = calculate_risk_adjusted_score(
            br['total_return_percent'], 
            br['mdd_percent'], 
            br['win_rate'], 
            br['num_trades']
        )
        
        # 샤프 비율 유사 점수
        sharpe_like = calculate_sharpe_like_ratio(
            br['total_return_percent'], 
            br['mdd_percent']
        )
        
        result['risk_adjusted_score'] = risk_adjusted_score
        result['sharpe_like_ratio'] = sharpe_like
    
    # 다양한 기준으로 정렬
    print("\n" + "="*80)
    print("1. 리스크 조정 점수 순위 (수익률-MDD*0.5+승률*0.1+거래수조정)")
    print("="*80)
    
    risk_sorted = sorted(valid_results, key=lambda x: x['risk_adjusted_score'], reverse=True)
    for i, result in enumerate(risk_sorted[:5]):
        br = result['backtest_results']
        params = result['params']
        print(f"\n{i+1}위: 리스크조정점수 {result['risk_adjusted_score']:.2f}")
        print(f"  수익률: {br['total_return_percent']:.2f}%")
        print(f"  MDD: {br['mdd_percent']:.2f}%")
        print(f"  승률: {br['win_rate']:.2f}%")
        print(f"  거래수: {br['num_trades']}")
        print(f"  손익비: {br['profit_factor']:.2f}")
        print(f"  주요 파라미터: seq_len={params['sequence_length']}, hidden_dim={params['hidden_dim']}, ")
        print(f"                 buy_thresh={params['signal_thresh_buy']}, sell_thresh={params['signal_thresh_sell']}")
    
    print("\n" + "="*80)
    print("2. 샤프 비율 유사 점수 순위 (수익률/|MDD|)")
    print("="*80)
    
    sharpe_sorted = sorted(valid_results, key=lambda x: x['sharpe_like_ratio'], reverse=True)
    for i, result in enumerate(sharpe_sorted[:5]):
        br = result['backtest_results']
        params = result['params']
        print(f"\n{i+1}위: 샤프비율유사 {result['sharpe_like_ratio']:.2f}")
        print(f"  수익률: {br['total_return_percent']:.2f}%")
        print(f"  MDD: {br['mdd_percent']:.2f}%")
        print(f"  승률: {br['win_rate']:.2f}%")
        print(f"  거래수: {br['num_trades']}")
        print(f"  손익비: {br['profit_factor']:.2f}")
        print(f"  주요 파라미터: seq_len={params['sequence_length']}, hidden_dim={params['hidden_dim']}, ")
        print(f"                 buy_thresh={params['signal_thresh_buy']}, sell_thresh={params['signal_thresh_sell']}")
    
    print("\n" + "="*80)
    print("3. MDD 낮은 순위 (안전성 중심)")
    print("="*80)
    
    # MDD가 낮고 수익률이 양수인 것들
    safe_results = [r for r in valid_results if r['backtest_results']['total_return_percent'] > 0]
    mdd_sorted = sorted(safe_results, key=lambda x: abs(x['backtest_results']['mdd_percent']))
    
    for i, result in enumerate(mdd_sorted[:5]):
        br = result['backtest_results']
        params = result['params']
        print(f"\n{i+1}위: MDD {br['mdd_percent']:.2f}%")
        print(f"  수익률: {br['total_return_percent']:.2f}%")
        print(f"  MDD: {br['mdd_percent']:.2f}%")
        print(f"  승률: {br['win_rate']:.2f}%")
        print(f"  거래수: {br['num_trades']}")
        print(f"  손익비: {br['profit_factor']:.2f}")
        print(f"  주요 파라미터: seq_len={params['sequence_length']}, hidden_dim={params['hidden_dim']}, ")
        print(f"                 buy_thresh={params['signal_thresh_buy']}, sell_thresh={params['signal_thresh_sell']}")
    
    print("\n" + "="*80)
    print("4. 고수익률 안정형 (수익률 30%+ & MDD -20% 이내)")
    print("="*80)
    
    # 수익률 30% 이상, MDD -20% 이내
    stable_high_return = [
        r for r in valid_results 
        if r['backtest_results']['total_return_percent'] >= 30 
        and r['backtest_results']['mdd_percent'] >= -20
    ]
    
    if stable_high_return:
        stable_sorted = sorted(stable_high_return, key=lambda x: x['risk_adjusted_score'], reverse=True)
        for i, result in enumerate(stable_sorted[:5]):
            br = result['backtest_results']
            params = result['params']
            print(f"\n{i+1}위: 리스크조정점수 {result['risk_adjusted_score']:.2f}")
            print(f"  수익률: {br['total_return_percent']:.2f}%")
            print(f"  MDD: {br['mdd_percent']:.2f}%")
            print(f"  승률: {br['win_rate']:.2f}%")
            print(f"  거래수: {br['num_trades']}")
            print(f"  손익비: {br['profit_factor']:.2f}")
            print(f"  주요 파라미터: seq_len={params['sequence_length']}, hidden_dim={params['hidden_dim']}, ")
            print(f"                 buy_thresh={params['signal_thresh_buy']}, sell_thresh={params['signal_thresh_sell']}")
    else:
        print("조건을 만족하는 결과가 없습니다.")
    
    print("\n" + "="*80)
    print("5. 종합 추천 (리스크 조정 점수 상위 5개)")
    print("="*80)
    
    # 최종 추천
    final_recommendations = risk_sorted[:5]
    for i, result in enumerate(final_recommendations):
        br = result['backtest_results']
        params = result['params']
        print(f"\n*** 추천 {i+1}위 ***")
        print(f"종합점수: {result['risk_adjusted_score']:.2f}")
        print(f"수익률: {br['total_return_percent']:.2f}%")
        print(f"MDD: {br['mdd_percent']:.2f}%")
        print(f"승률: {br['win_rate']:.2f}%")
        print(f"거래수: {br['num_trades']}")
        print(f"손익비: {br['profit_factor']:.2f}")
        print(f"최적 파라미터:")
        for key, value in params.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    analyze_results() 