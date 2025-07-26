import pandas as pd

def analyze_performance(prices: list, base_date: str, days_ahead: int = 5) -> dict:
    """
    base_date 이후의 주가 데이터를 바탕으로 평균 상승률, 최고 상승률, 거래량 증감률을 계산.

    Args:
        prices (list): 주가 정보가 담긴 딕셔너리 리스트. (각 딕셔너리에 'date', 'open', 'close', 'volume' 포함)
        base_date (str): 기준 날짜 (예: '2025-06-13')
        days_ahead (int): 분석 대상일 수 (기준일 이후 몇 일치 데이터를 분석할지)

    Returns:
        dict: 분석 결과 (평균 상승률, 최고 상승률, 거래량 증감률 등)
    """
    # --- 1. 날짜 정렬 및 DataFrame화 ---
    df = pd.DataFrame(prices).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 숫자 컬럼을 명시적으로 float/int로 변환
    for col in ["open", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])  # 날짜 파싱 실패 제거
    df = df.sort_values("date").reset_index(drop=True)

    # --- 2. base_date 위치 찾기 ---
    base_dt = pd.to_datetime(base_date)
    if base_dt not in df["date"].values:
        return None  # base_date가 price 안에 없을 경우 분석 불가

    base_idx = df.index[df["date"] == base_dt][0]

    # --- 3. 이후 days_ahead만큼의 유효한 주가 데이터 확보 ---
    future_df = df.iloc[base_idx + 1 : base_idx + 1 + days_ahead]
    if future_df.empty:
        return None

    # --- 4. 상승률 계산 ---
    base_close = df.loc[base_idx, "close"]
    future_df["pct_change"] = (future_df["close"] - base_close) / base_close * 100

    avg_pct_change = future_df["pct_change"].mean()
    max_pct_change = future_df["pct_change"].max()

    # --- 5. 거래량 증감률 계산 (base vs future 평균) ---
    base_volume = df.loc[base_idx, "volume"]
    future_volume_avg = future_df["volume"].mean()
    if base_volume == 0:
        volume_change_pct = None
    else:
        volume_change_pct = (future_volume_avg - base_volume) / base_volume * 100

    return {
        "avg_pct_change": round(avg_pct_change, 2),
        "max_pct_change": round(max_pct_change, 2),
        "volume_change_pct": round(volume_change_pct, 2) if volume_change_pct is not None else None,
    }


def analyze_before_after_performance(prices: list, base_date: str, days_before: int = 5, days_after: int = 5) -> dict:
    """
    base_date를 기준으로 간단한 3개 지표만 계산합니다.
    1. 평균 상승률: 이후일 주가 평균 / 이전일 주가 평균 * 100
    2. 최고 상승률: 이후일 최고 주가 / 이전일 주가 평균 * 100  
    3. 거래량 증감률: 이후일 거래량 평균 / 이전일 거래량 평균 * 100
    
    Args:
        prices (list): 주가 정보가 담긴 딕셔너리 리스트
        base_date (str): 기준 날짜
        days_before (int): 기준일 이전 분석 대상일 수
        days_after (int): 기준일 이후 분석 대상일 수
    
    Returns:
        dict: 3개 핵심 지표만 포함한 분석 결과
    """
    # --- 1. 데이터 전처리 ---
    df = pd.DataFrame(prices).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # 숫자 컬럼 변환
    for col in ["close", "high", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # --- 2. base_date 찾기 ---
    base_dt = pd.to_datetime(base_date)
    if base_dt not in df["date"].values:
        return {"error": f"기준 날짜 {base_date}가 데이터에 없습니다."}
    
    base_idx = df.index[df["date"] == base_dt][0]
    
    # --- 3. 이전/이후 기간 데이터 분리 ---
    before_start_idx = max(0, base_idx - days_before)
    before_df = df.iloc[before_start_idx:base_idx]  # 기준일 제외
    
    after_end_idx = min(len(df), base_idx + 1 + days_after)
    after_df = df.iloc[base_idx + 1:after_end_idx]  # 기준일 다음날부터
    
    # --- 4. 데이터 충분성 검사 ---
    if before_df.empty or after_df.empty:
        return {"error": "이전 또는 이후 데이터가 충분하지 않습니다."}
    
    # --- 5. 핵심 지표 3개 계산 ---
    # 이전일 평균값들
    before_avg_close = before_df["close"].mean()
    before_avg_volume = before_df["volume"].mean()
    
    # 이후일 평균/최고값들  
    after_avg_close = after_df["close"].mean()
    after_max_close = after_df["close"].max()
    after_avg_volume = after_df["volume"].mean()
    
    # 변화율 계산 (기존 비율에서 100을 빼서 실제 상승률로 변경)
    avg_price_change = ((after_avg_close / before_avg_close) - 1) * 100 if before_avg_close > 0 else 0
    max_price_change = ((after_max_close / before_avg_close) - 1) * 100 if before_avg_close > 0 else 0
    volume_change = ((after_avg_volume / before_avg_volume) - 1) * 100 if before_avg_volume > 0 else 0
    
    return {
        "avg_price_change": round(avg_price_change, 2),
        "max_price_change": round(max_price_change, 2), 
        "volume_change": round(volume_change, 2)
    }


def format_analysis_summary(analysis_result: dict) -> str:
    """
    간단한 3개 지표만 표시하는 요약 포맷
    """
    if "error" in analysis_result:
        return f"❌ 분석 오류: {analysis_result['error']}"
    
    summary = "📊 **핵심 지표 분석**\n"
    summary += f"• 평균 상승률: {analysis_result['avg_price_change']:+.1f}%\n"
    summary += f"• 최고 상승률: {analysis_result['max_price_change']:+.1f}%\n" 
    summary += f"• 거래량 증감률: {analysis_result['volume_change']:+.1f}%"
    
    return summary