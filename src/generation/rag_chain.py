import ollama

def generate_answer(question, context, model="phi:2.7b"):
    """Generate answer using RAG"""
    prompt = f"""Answer based ONLY on this context:

CONTEXT: {context}

QUESTION: {question}

ANSWER:"""
    
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]