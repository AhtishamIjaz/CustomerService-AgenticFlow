import streamlit as st
import uuid
import sqlite3
from engine.graph import graph_app
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Pro Customer Agent", page_icon="🤖")

# --- DATABASE UTILITY ---
def get_all_threads():
    """Fetch unique thread IDs from the LangGraph checkpoints database."""
    try:
        conn = sqlite3.connect("state/checkpoints.db")
        cursor = conn.cursor()
        # LangGraph stores thread_id in the checkpoints table
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        threads = [row[0] for row in cursor.fetchall()]
        conn.close()
        return threads
    except:
        return []

# --- SIDEBAR: HISTORY MANAGEMENT ---
with st.sidebar:
    st.title("📜 Chat History")
    
    # 1. Option to start a new chat
    if st.button("+ New Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    
    # 2. List old threads
    st.subheader("Previous Sessions")
    past_threads = get_all_threads()
    
    for tid in past_threads:
        if st.button(f"Chat: {tid[:8]}...", key=tid, use_container_width=True):
            st.session_state.thread_id = tid
            # When switching threads, we clear the UI state so LangGraph can reload it
            st.session_state.messages = [] 
            st.rerun()

# --- MAIN CHAT LOGIC ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

st.title("Industrial Customer Support")
st.info(f"Active Thread: {st.session_state.thread_id}")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fetch history from LangGraph if UI state is empty (e.g., after refresh/switch)
if not st.session_state.messages:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state = graph_app.get_state(config)
    if state.values.get("messages"):
        for m in state.values["messages"]:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            st.session_state.messages.append({"role": role, "content": m.content})

# Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        # Using token streaming
        for msg, metadata in graph_app.stream(
            {"messages": [HumanMessage(content=prompt)]}, 
            config=config, 
            stream_mode="messages"
        ):
            if metadata.get("langgraph_node") == "generator":
                full_response += msg.content
                placeholder.markdown(full_response + "▌")
        
        placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})