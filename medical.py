import streamlit as st
import ollama
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Page setup
st.set_page_config(page_title="Dr. Medibot", page_icon="🩺", layout="wide")

# ==================== VERY SIMPLE CSS ====================
st.markdown("""
<style>
    /* Simple message bubbles */
    .user-msg {
        background-color: #0a5c3f;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: right;
    }
    .doctor-msg {
        background-color: #e0e0e0;
        color: black;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #0a5c3f;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs_ready" not in st.session_state:
    st.session_state.docs_ready = False

# ==================== HEADER ====================
st.title("🩺 Dr. Medibot")
st.caption("Ask me questions based on your medical documents")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.subheader("📄 Upload Medical PDFs")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing..."):
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Split
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            
            # Create vector store
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
            st.session_state.docs_ready = True
            
            # Cleanup
            os.unlink(tmp_path)
            
            st.success(f"✅ Ready! {len(chunks)} sections loaded")
            st.balloons()
    
    st.divider()
    
    if st.session_state.docs_ready:
        st.success("📚 Documents loaded")
    else:
        st.info("📚 No documents yet")
    
    model = st.selectbox("AI Model", ["phi:2.7b", "tinyllama:1.1b"], index=0)

# ==================== CHAT AREA ====================
# Show chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">👤 <strong>You:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="doctor-msg">🩺 <strong>Dr. Medibot:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)

# Input
if st.session_state.docs_ready:
    user_input = st.chat_input("Ask a question...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Search and respond
        with st.spinner("Thinking..."):
            try:
                # Search documents
                results = st.session_state.vectorstore.similarity_search(user_input, k=3)
                
                if results:
                    context = "\n\n".join([r.page_content for r in results])
                    
                    prompt = f"""Answer this medical question using ONLY the context below.

CONTEXT:
{context}

QUESTION: {user_input}

ANSWER (short, helpful, based only on context):"""
                    
                    response = ollama.chat(
                        model=model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    answer = response["message"]["content"]
                    answer += f"\n\n*(From your document)*"
                    
                else:
                    answer = "I couldn't find that in your document. Please try different words."
                    
            except Exception as e:
                answer = f"Error: {str(e)}"
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
else:
    st.info("👈 Please upload a medical PDF file in the sidebar to start")

# Footer
st.divider()
st.caption("⚠️ For educational purposes. Consult a real doctor.")