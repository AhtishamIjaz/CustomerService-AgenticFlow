import sqlite3
import operator
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Import your nodes
from engine.nodes import retrieve_node, grade_documents_node, generate_node

# --- 1. Define the Graph State ---
class AgentState(TypedDict):
    # Annotated with operator.add allows messages to be appended automatically
    messages: Annotated[List, operator.add]
    context: str
    relevance: str  # Stores 'yes' or 'no' from the grader

# --- 2. Initialize the State Graph ---
workflow = StateGraph(AgentState)

# --- 3. Add Nodes ---
workflow.add_node("retriever", retrieve_node)
workflow.add_node("grader", grade_documents_node)
workflow.add_node("generator", generate_node)

# --- 4. Define the Logic Flow (Edges) ---
workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "grader")

# Conditional Edge Logic
# This routes the flow based on the 'relevance' score from the grader
workflow.add_conditional_edges(
    "grader",
    lambda state: state["relevance"],
    {
        "yes": "generator",
        "no": "generator" # Both lead to generator, but generator handles the logic
    }
)

workflow.add_edge("generator", END)

# --- 5. Add Persistence (The Industrial Fix) ---
# We manually create the connection to avoid the GeneratorContextManager error.
# 'check_same_thread=False' is vital for Streamlit's multi-threaded nature.
def compile_graph():
    db_path = "state/checkpoints.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)

# This is the object your app.py imports
graph_app = compile_graph()