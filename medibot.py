import streamlit as st
import ollama

st.set_page_config(page_title="Dr. Medibot", page_icon="🩺", layout="wide")

# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================== HEADER ====================
st.title("🩺 Dr. Medibot")
st.caption("Your Friendly Doctor • I Listen, I Ask, I Help")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### Hello! I'm your personal doctor.")
    st.markdown("I don't just give answers - **I listen and ask questions** like a real doctor would.")
    
    st.markdown("---")
    
    st.markdown("### How I work:")
    st.markdown("""
    1️⃣ You tell me your symptom  
    2️⃣ I ask follow-up questions  
    3️⃣ Based on your answers, I analyze  
    4️⃣ I give you my honest opinion  
    """)
    
    st.markdown("---")
    
    st.markdown("### I can help with:")
    st.markdown("""
    - 🤒 Symptoms analysis  
    - 🏠 Home remedies  
    - ⚠️ Warning signs  
    - 💙 General health advice  
    """)
    
    st.markdown("---")
    st.warning("⚠️ **Important:** If symptoms become severe, please see a real doctor.")
    
    model = st.selectbox("🧠 AI Model", ["phi:2.7b", "tinyllama:1.1b"], index=0)

# ==================== WELCOME MESSAGE (only when no messages) ====================
if len(st.session_state.messages) == 0:
    st.info("""
    **💙 How I work:**
    
    1️⃣ Tell me your symptom - Example: "I have a headache"  
    2️⃣ I'll ask questions - Like a real doctor would  
    3️⃣ You answer - The more details, the better  
    4️⃣ I'll analyze - Based on medical knowledge  
    5️⃣ I'll advise you - Home care & when to see a doctor  
    
    ---
    ✨ **Ready? Tell me what's bothering you...**
    """)

# ==================== CHAT DISPLAY ====================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# ==================== CHAT INPUT ====================
user_input = st.chat_input("What's bothering you? Example: 'I have a headache'...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🩺 Thinking..."):
            try:
                # Get conversation history
                history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-8:]])
                exchange_count = len(st.session_state.messages) // 2
                
                if exchange_count <= 2:
                    # STAGE 1: Ask questions first
                    prompt = f"""You are Dr. Medibot, a caring doctor. 

Conversation history:
{history}

Rules:
- Be warm and caring
- Ask ONE specific question to understand better
- Do NOT give diagnosis or advice yet
- Keep it short

Your response:"""
                else:
                    # STAGE 2: Give short bullet point advice
                    prompt = f"""You are Dr. Medibot. Based on this conversation:

{history}

Give a SHORT response using EXACTLY this format:

**Possible causes:**
- (cause 1)
- (cause 2)

**What you can do at home:**
- (action 1)
- (action 2)
- (action 3)

**See a doctor if:**
- (warning 1)
- (warning 2)

Keep it very short. Use simple words. Be warm but brief."""
                
                response_obj = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                response = response_obj["message"]["content"]
                
                # Clean up any robotic phrases
                response = response.replace("As an AI", "As your doctor")
                response = response.replace("I am an AI", "I am Dr. Medibot")
                
            except Exception as e:
                response = f"⚠️ Error: {str(e)}\n\nMake sure Ollama is running with `ollama serve` in another terminal."
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ==================== CLEAR BUTTON ====================
if len(st.session_state.messages) > 0:
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Start New Consultation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ==================== FOOTER ====================
st.divider()
st.caption("💙 Dr. Medibot | If symptoms become severe, please consult a real doctor.")