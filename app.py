# app.py

import streamlit as st
from retriever import search
from generation import prompt_and_generate

st.title("🤖 투자 어시스턴트")

# 세션 상태에 메시지 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 내역 출력
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# 사용자 입력 처리
if prompt := st.chat_input("궁금한 점을 물어보세요."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 검색 함수(search)를 전달해 prompt_and_generate 호출
    response = prompt_and_generate(prompt.strip(), search)
    full_response = f"bot: {response}"
    
    with st.chat_message("assistant"):
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
