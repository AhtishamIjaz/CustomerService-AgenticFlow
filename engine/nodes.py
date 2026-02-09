import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# 1. Initialize the LLM
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. Setup Retrieval (FAISS)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- NODES ---

def retrieve_node(state):
    """Step 1: Retrieve context based on the user's latest query."""
    messages = state["messages"]
    last_message = messages[-1].content
    docs = retriever.invoke(last_message)
    context = "\n".join([doc.page_content for doc in docs])
    return {"context": context}

def grade_documents_node(state):
    """
    Step 2: INDUSTRIAL GRADING. 
    Fixed to handle cases where the LLM returns "1. 'yes'" instead of just "yes".
    """
    question = state["messages"][-1].content
    context = state.get("context", "")

    grade_prompt = (
        f"You are a Quality Control Analyst for Tech-Pro Solutions.\n"
        f"USER QUESTION: {question}\n"
        f"RETRIEVED CONTEXT: {context}\n\n"
        "TASK: Evaluate if the context is useful. Return ONLY one word:\n"
        "- 'yes' (answer is there)\n"
        "- 'maybe' (partially related)\n"
        "- 'no' (irrelevant)\n"
        "Your decision:"
    )
    
    response = llm.invoke([HumanMessage(content=grade_prompt)])
    raw_content = response.content.lower()

    # LOGIC FIX: String Keyword Matching
    # This prevents KeyError by ensuring 'decision' is exactly what the graph expects.
    if "yes" in raw_content:
        decision = "yes"
    elif "maybe" in raw_content:
        decision = "maybe"
    else:
        decision = "no"

    return {"relevance": decision}

def generate_node(state):
    """
    Step 3: ADAPTIVE GENERATION.
    Handles different relevance levels with industrial-grade safety.
    """
    messages = state["messages"]
    context = state.get("context", "")
    relevance = state.get("relevance", "yes")

    # INDUSTRIAL FALLBACK: If 'no', provide contact info immediately.
    if relevance == "no":
        return {"messages": [AIMessage(content=(
            "I'm sorry, I don't have that specific information in my knowledge base. "
            "However, you can reach our human support team 24/7 at ahtishamijaz55@gmail.com "
            "or call 1-800-TECH-PRO (Mon-Fri, 9AM-6PM EST)."
        ))]}

    # ADAPTIVE SYSTEM PROMPT
    system_base = (
        "You are a professional Tech-Pro Customer Support Expert. "
        "Instructions:\n"
        "1. Answer ONLY using the provided context.\n"
        "2. If the user asks for a detail not explicitly listed, explain what is available instead of guessing.\n"
        "3. Keep answers concise (2-3 lines).\n"
    )

    if relevance == "maybe":
        system_base += "NOTE: The information might be partially related. Be honest about what you find."

    full_prompt = SystemMessage(content=f"{system_base}\n\nCONTEXT:\n{context}")
    
    response = llm.invoke([full_prompt] + messages)
    return {"messages": [response]}