import os
import sqlite3
import operator
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
# Updated Import Path
from langgraph.checkpoint.sqlite import SqliteSaver

# Import your nodes
from engine.nodes import retrieve_node, grade_documents_node, generate_node

# --- 1. Define the Graph State ---
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    context: str
    relevance: str  

# --- 2. Initialize the State Graph ---
workflow = StateGraph(AgentState)

# --- 3. Add Nodes ---
workflow.add_node("retriever", retrieve_node)
workflow.add_node("grader", grade_documents_node)
workflow.add_node("generator", generate_node)

# --- 4. Define the Logic Flow (Edges) ---
workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "grader")

workflow.add_conditional_edges(
    "grader",
    lambda state: state["relevance"],
    {
        "yes": "generator",
        "no": "generator" 
    }
)

workflow.add_edge("generator", END)

# --- 5. Add Persistence ---
def compile_graph():
    # Ensure the 'state' directory exists in the container
    if not os.path.exists("state"):
        os.makedirs("state")
        
    db_path = "state/checkpoints.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    # The modern SqliteSaver expects a connection
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)

# This is the object your app.py imports
graph_app = compile_graph()