import streamlit as st
from pymilvus import connections
import time
import os

from utils.setup import setup_executors
from db.document_store import store_documents
from rag import answer_question

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

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("데이터 관리")
    st.markdown("새로운 `.txt` 파일을 `data` 폴더에 추가한 후, 아래 버튼을 눌러 데이터베이스를 업데이트하세요.")
    if st.button("데이터베이스 초기화 및 문서 처리"):
        with st.spinner("문서 처리 중... 기존 데이터를 삭제하고 새로 임베딩합니다."):
            store_documents(segmentation_executor, embedding_executor)
            st.success("문서 처리가 완료되었습니다!")
            st.info("채팅 기록이 초기화됩니다.")
            st.session_state.messages = [] 
            time.sleep(2)
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("뉴스 기사에 대해 질문해주세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중입니다..."):
            answer, reference = answer_question(prompt, embedding_executor, completion_executor)
            
            response = f"{answer}\n\n"
            if reference:
                response += "**--- 참고 자료 ---**\n"
                sources = list(dict.fromkeys([ref['source'] for ref in reference]))
                for src in sources:
                    response += f"- {src}\n"
            
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response}) 