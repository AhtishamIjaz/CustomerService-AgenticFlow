import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_knowledge_base():
    # 1. Load your raw text file
    # Make sure you have a file at data/knowledge_base.txt
    loader = TextLoader("data/knowledge_base.txt")
    documents = loader.load()

    # 2. Split text into manageable chunks
    # We use 500 characters with a 50-character overlap for context continuity
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # 3. Create Embeddings (The mathematical "fingerprints" of your text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create and Save the FAISS Index
    print("Creating vector database... this may take a moment.")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save it to the folder your engine/nodes.py is looking for
    vectorstore.save_local("faiss_index")
    print("✅ Success! 'faiss_index' folder created.")

if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists("data/knowledge_base.txt"):
        print("❌ Error: data/knowledge_base.txt not found!")
    else:
        create_knowledge_base()