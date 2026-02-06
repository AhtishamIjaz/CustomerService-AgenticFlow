import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize the embedding model
# 'all-MiniLM-L6-v2' is fast and great for customer service tasks
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_or_load_vector_store(data_path="data/"):
    """
    Scans the data folder, processes documents, and creates a FAISS index.
    """
    documents = []
    
    # Industrial practice: Support both PDF and Text files
    for file in os.listdir(data_path):
        full_path = os.path.join(data_path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(full_path)
            documents.extend(loader.load())

    if not documents:
        print("No documents found in data/ folder. Please add a PDF or TXT file.")
        return None

    # Splitting logic: 1000 characters with a small overlap to keep context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)

    # Create and return the retriever
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 3})

# Helper to get the retriever instance
retriever = build_or_load_vector_store()