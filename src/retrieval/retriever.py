def get_retriever(vector_store, k=4):
    """Create retriever for RAG"""
    return vector_store.as_retriever(search_kwargs={"k": k})