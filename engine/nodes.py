import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# 1. Initialize the LLM (Using the newest Llama 3.3 model)
llm = ChatGroq(
    temperature=0, # Set to 0 for Industrial reliability (no random guessing)
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. Setup Retrieval (FAISS)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Ensure the folder 'faiss_index' exists from your initialization script
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- NODES ---

def retrieve_node(state):
    """
    Step 1: Get documents based on the user's question.
    """
    messages = state["messages"]
    last_message = messages[-1].content
    
    # Retrieve relevant docs
    docs = retriever.invoke(last_message)
    context = "\n".join([doc.page_content for doc in docs])
    
    return {"context": context}

def grade_documents_node(state):
    """
    Step 2: INDUSTRIAL GRADE - Check if the info found is actually relevant.
    This prevents the AI from answering out-of-scope questions.
    """
    messages = state["messages"]
    question = messages[-1].content
    context = state.get("context", "")

    grade_prompt = (
        f"You are a Quality Control grader. Analyze if the following context "
        f"contains the answer to this question: '{question}'\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Give a binary score: 'yes' if it's relevant, or 'no' if it's not. "
        "Just say the word 'yes' or 'no' and nothing else."
    )
    
    response = llm.invoke([HumanMessage(content=grade_prompt)])
    decision = response.content.lower().strip()
    
    # We pass the decision to the graph to choose the next node
    return {"relevance": decision}

def generate_node(state):
    """
    Step 3: Generate a precise 2-3 line answer.
    """
    messages = state["messages"]
    context = state.get("context", "No context provided.")
    relevance = state.get("relevance", "yes")

    if relevance == "no":
        return {"messages": [AIMessage(content="I'm sorry, I don't have information about that in my database. Is there anything else I can help with?")]}

    system_prompt = SystemMessage(content=(
        "You are a professional Customer Service Assistant. "
        "Rules:\n"
        "1. Answer ONLY using the provided context.\n"
        "2. Keep the answer strictly between 2 to 3 lines.\n"
        "3. If the answer is not in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context}"
    ))
    
    response = llm.invoke([system_prompt] + messages)
    return {"messages": [response]}