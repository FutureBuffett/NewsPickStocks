import streamlit as st
from pymilvus import connections
import time
import os
import diskcache

from utils.setup import setup_executors
from db.document_store import store_documents
from rag import answer_question, hybrid_search
from utils.chunk_filter import is_irrelevant_chunk

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
        segmented_chunks = segmentation_executor.execute({"text": prompt})
        filtered_chunks = []
        for chunk in segmented_chunks:
            chunk_text = chunk if isinstance(chunk, str) else ' '.join(chunk)
            if not is_irrelevant_chunk(chunk_text):
                filtered_chunks.append(chunk_text)
        filtered_full_text = '\n'.join(filtered_chunks)

        ranked_stocks = hybrid_search(
            prompt, segmentation_executor, embedding_executor,
            filtered_full_text=filtered_full_text, filtered_chunks=filtered_chunks
        )
        print("순위" + str(ranked_stocks))
        if isinstance(ranked_stocks, str):
            response = ranked_stocks
        else:
            response = "**추천 종목 랭킹:**\n"
            for stock, score in ranked_stocks:
                response += f"- {stock}: {score:.4f}\n"
        st.markdown(response) 