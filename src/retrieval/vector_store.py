from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(chunks, persist_dir="medibot_db"):
    """Create vector database"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    print(f"✅ Vector store created")
    return vector_store

def search_similar(vector_store, query, k=3):
    """Search for similar documents"""
    results = vector_store.similarity_search(query, k=k)
    print(f"🔍 Found {len(results)} relevant sections")
    return results