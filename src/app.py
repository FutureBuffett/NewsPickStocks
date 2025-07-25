import streamlit as st
from pymilvus import connections
import time
import os
import diskcache
import pandas as pd

from utils.setup import setup_executors
from db.document_store import store_documents
from rag import answer_question, hybrid_search
from utils.chunk_filter import is_irrelevant_chunk
from utils.clean_text import clean_text
from utils.stock_analysis import analyze_performance, analyze_before_after_performance, format_analysis_summary
import plotly.graph_objects as go
import plotly.express as px

# --- 페이지 설정 ---
st.set_page_config(page_title="뉴스 기반 주식 추천", page_icon="📰")
st.title("📰 뉴스 기반 주식 추천")

# --- 초기화 ---
@st.cache_resource
def init_executors_and_db():
    try:
        executors = setup_executors()
        connections.connect()
        return executors
    except Exception as e:
        st.error(f"초기화 중 오류 발생: {e}")
        return None

executors = init_executors_and_db()
if executors is None:
    st.stop()

segmentation_executor, embedding_executor, completion_executor = executors

# --- 사이드바 기능 ---
with st.sidebar:
    st.header("데이터 관리")
    st.markdown("새로운 `.txt` 파일을 `data` 폴더에 추가한 후, 아래 버튼을 눌러 데이터베이스를 업데이트하세요.")
    if st.button("데이터베이스 초기화 및 문서 처리"):
        with st.spinner("문서 처리 중... 기존 데이터를 삭제하고 새로 임베딩합니다."):
            store_documents(segmentation_executor, embedding_executor, data_dir="data")
            st.success("문서 처리가 완료되었습니다!")
            st.info("채팅 기록이 초기화됩니다.")
            time.sleep(2)
            st.rerun()
    if st.button("임베딩/청킹 캐시 삭제"):
        seg_cache = diskcache.Cache('segmentation_cache.db')
        emb_cache = diskcache.Cache('embedding_cache.db')
        seg_cache.clear()
        emb_cache.clear()
        st.success("임베딩/청킹 캐시가 모두 삭제되었습니다.")

# --- 사용자 입력 및 추천 ---
prompt = st.text_area("뉴스 기사 입력", "", height=200)

if st.button("추천 종목 분석하기") and prompt.strip():
    with st.spinner("추천 종목을 분석하는 중입니다..."):
        cleaned_prompt = clean_text(prompt)
        segmented_chunks = segmentation_executor.execute({"text": prompt})
        filtered_chunks = []
        for chunk in segmented_chunks:
            chunk_text = chunk if isinstance(chunk, str) else ' '.join(chunk)
            if not is_irrelevant_chunk(chunk_text):
                filtered_chunks.append(chunk_text)
        filtered_full_text = '\n'.join(filtered_chunks)

        ranked_stocks = hybrid_search(
            cleaned_prompt,
            segmentation_executor,
            embedding_executor,
            filtered_full_text=filtered_full_text,
            filtered_chunks=filtered_chunks
        )
        for comp in ranked_stocks:
            result = analyze_performance(comp["prices"], comp["base_date"])
            if result:
                comp.update(result)

        # 출력 처리
        if isinstance(ranked_stocks, str):
            st.markdown(ranked_stocks)
        elif not ranked_stocks:
            st.warning("추천된 종목이 없습니다. 뉴스 내용이 너무 일반적이거나 관련 데이터가 부족할 수 있습니다.")
        else:
            st.markdown("## 🏆 추천 종목 랭킹")
            
            for idx, item in enumerate(ranked_stocks, 1):
                st.markdown(f"### {idx}. 📊 {item['company']} ({item['base_date']} 기준)")
                
                # 메인 레이아웃: 왼쪽에 랭킹 정보, 오른쪽에 핵심 지표
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**📈 종목 정보**")
                    st.markdown(f"- **유사도 점수**: {item['score']:.4f}")
                    st.markdown(f"- **기준 날짜**: {item['base_date']}")
                    st.markdown(f"- **데이터 수**: {len(item['prices'])}일")
                
                with col2:
                    # 간단한 3개 지표 분석 실행
                    detailed_analysis = analyze_before_after_performance(
                        item["prices"], 
                        item["base_date"], 
                        days_before=5, 
                        days_after=5
                    )
                    
                    # 분석 결과 표시
                    if detailed_analysis and "error" not in detailed_analysis:
                        st.markdown("**📊 핵심 지표 분석**")
                        st.markdown(f"• **평균 상승률**: {detailed_analysis['avg_price_change']:+.1f}%")
                        st.markdown(f"• **최고 상승률**: {detailed_analysis['max_price_change']:+.1f}%")
                        st.markdown(f"• **거래량 증감률**: {detailed_analysis['volume_change']:+.1f}%")
                    else:
                        st.markdown("**⚠️ 분석 불가**")
                        st.markdown(f"오류: {detailed_analysis.get('error', '알 수 없는 오류')}")
                
                # 주가 데이터 테이블과 상세 분석
                st.markdown("---")
                
                # 주가 데이터 테이블
                prices_df = pd.DataFrame(item["prices"]).sort_values("date")
                st.markdown("**📋 주가 데이터**")
                # st.dataframe(prices_df, use_container_width=True)
                
                # 상세 분석 결과 (확장 가능한 섹션)
                with st.expander("📋 상세 분석 데이터 보기"):
                    if detailed_analysis and "error" not in detailed_analysis:
                        st.json(detailed_analysis)
                    else:
                        st.error("상세 분석 데이터를 불러올 수 없습니다.")
                    
                    # 주가 차트
                    if len(prices_df) > 1:
                        # 차트 데이터 준비
                        prices_df_chart = prices_df.copy()
                        prices_df_chart['date'] = pd.to_datetime(prices_df_chart['date'])
                        base_dt = pd.to_datetime(item['base_date'])
                        
                        # 이전/이후/기준일 구분
                        prices_df_chart['period'] = prices_df_chart['date'].apply(
                            lambda x: '이전' if x < base_dt else ('기준일' if x == base_dt else '이후')
                        )
                        
                        # Plotly로 차트 생성
                        fig = go.Figure()
                        
                        # 색상 구분
                        colors = {'이전': '#FF6B6B', '기준일': '#4ECDC4', '이후': '#45B7D1'}
                        
                        for period in ['이전', '기준일', '이후']:
                            period_data = prices_df_chart[prices_df_chart['period'] == period]
                            if not period_data.empty:
                                fig.add_trace(go.Scatter(
                                    x=period_data['date'],
                                    y=period_data['close'],
                                    mode='lines+markers',
                                    name=period,
                                    line=dict(color=colors[period], width=3),
                                    marker=dict(size=8)
                                ))
                        
                        # 기준일에 수직선 추가
                        base_dt_for_plot = base_dt.to_pydatetime() if hasattr(base_dt, 'to_pydatetime') else base_dt
                        
                        fig.add_shape(
                            type="line",
                            x0=base_dt_for_plot,
                            y0=0,
                            x1=base_dt_for_plot,
                            y1=1,
                            yref="paper",
                            line=dict(color="gray", width=2, dash="dash")
                        )
                        
                        # 기준일 텍스트 annotation
                        fig.add_annotation(
                            x=base_dt_for_plot,
                            y=1.02,
                            yref="paper",
                            text="기준일",
                            showarrow=False,
                            font=dict(color="gray", size=12),
                            xanchor="center"
                        )
                        
                        # 차트 레이아웃 설정
                        fig.update_layout(
                            title=f"{item['company']} 주가 추이 ({item['base_date']} 기준)",
                            xaxis_title="날짜",
                            yaxis_title="종가 (원)",
                            height=400,
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        fig.update_yaxes(autorange=True)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("차트를 그리기에 데이터가 충분하지 않습니다.")
                
                # 구분선 (마지막 항목이 아닌 경우만)
                if idx < len(ranked_stocks):
                    st.markdown("---")
                    st.markdown("")  # 공백 추가


            # (선택) 디버깅용
            # st.json(ranked_stocks)
