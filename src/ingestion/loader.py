from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf(file_path):
    """Load PDF and extract text"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")
    return documents