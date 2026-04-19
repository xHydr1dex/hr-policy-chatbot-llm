# capstone_streamlit.py - NovaTech HR Assistant (Aria)
# Run: streamlit run capstone_streamlit.py
import streamlit as st
import uuid
from agent import build_agent, DOCUMENTS

DOMAIN_NAME  = "NovaTech HR Assistant"
KB_TOPICS    = [d["topic"] for d in DOCUMENTS]

st.set_page_config(page_title=DOMAIN_NAME, page_icon="🏢", layout="centered")
st.title("🏢 " + DOMAIN_NAME)
st.caption("Hi, I am Aria — your 24/7 HR policy companion at NovaTech Solutions.")

@st.cache_resource
def load_agent():
    return build_agent()

try:
    agent_app, embedder, collection = load_agent()
    st.success(f"Knowledge base loaded — {collection.count()} HR policy documents")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

with st.sidebar:
    st.header("About Aria")
    st.write(
        "Aria answers questions about NovaTech HR policies, benefits, leave rules, "
        "payroll, appraisals, and more — grounded strictly in the company handbook."
    )
    st.write(f"Session: {st.session_state.thread_id}")
    st.divider()
    st.write("**HR Topics Covered:**")
    for t in KB_TOPICS:
        st.write(f"• {t}")
    st.divider()
    if st.button("New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask me anything about NovaTech HR policies..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Checking HR policies..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not answer. Please contact hr@novatech.in.")
        st.write(answer)
        faith   = result.get("faithfulness", 0.0)
        sources = result.get("sources", [])
        if sources:
            st.caption(f"Sources: {', '.join(sources)} | Faithfulness: {faith:.2f}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
