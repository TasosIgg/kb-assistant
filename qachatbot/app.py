# app.py

import os

import streamlit as st
import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/ask")
API_KEY = os.environ.get("API_KEY", "dev-local-key")

st.set_page_config(page_title="KB Assistant")

st.title("Employee Self-Service Assistant")
st.caption(
    "Ask about onboarding, benefits, time off, learning & development, and remote-work "
    "policy. Answers are grounded in GitLab's public company handbook via retrieval-"
    "augmented generation (RAG) with a locally-run LLM."
)

EXAMPLE_QUESTIONS = [
    "How much paid time off do new employees get?",
    "What is GitLab's parental leave policy?",
    "How does onboarding work for new hires?",
    "What is the all-remote culture at GitLab?",
]

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question", "answer", "sources"}

st.write("Example questions:")
cols = st.columns(len(EXAMPLE_QUESTIONS))
example_clicked = None
for col, q in zip(cols, EXAMPLE_QUESTIONS):
    if col.button(q, use_container_width=True):
        example_clicked = q

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["sources"]:
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.markdown(f"- [{s['title']}]({s['url']})")

question = example_clicked or st.chat_input("Ask a question about the knowledge base...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer — can take a couple of minutes on this free CPU-only instance."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "question": question,
                        "history": [
                            {"question": t["question"], "answer": t["answer"]}
                            for t in st.session_state.history
                        ],
                    },
                    headers={"X-API-Key": API_KEY},
                    timeout=280,
                )

                if response.status_code == 200:
                    data = response.json()
                    st.write(data["answer"])
                    if data["sources"]:
                        with st.expander("Sources"):
                            for s in data["sources"]:
                                st.markdown(f"- [{s['title']}]({s['url']})")
                    st.caption(f"Answered in {data['latency_ms']} ms")

                    st.session_state.history.append({
                        "question": question,
                        "answer": data["answer"],
                        "sources": data["sources"],
                    })
                else:
                    st.error(f"API Error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "API is not running. Start it with:\n\n`uvicorn api:app --reload`"
                )
