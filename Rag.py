import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
from PyPDF2 import PdfReader

# Configure page
st.set_page_config(page_title="RAG with Gemini", page_icon="🤖", layout="wide")

# Initialize session state
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None

def load_embedding_model():
    """Load the sentence transformer model for embeddings"""
    if st.session_state.embeddings_model is None:
        with st.spinner("Loading embedding model..."):
            st.session_state.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
    return st.session_state.embeddings_model

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_txt(txt_file):
    """Extract text from uploaded TXT file"""
    return txt_file.read().decode('utf-8')

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def create_embeddings(chunks, embedding_model):
    """Create embeddings from text chunks using PyTorch tensors"""
    with st.spinner("Creating embeddings..."):
        embeddings = embedding_model.encode(
            chunks, 
            show_progress_bar=True, 
            convert_to_tensor=True
        )
    return embeddings

def retrieve_relevant_chunks(query, embeddings, chunks, embedding_model, k=3):
    """Retrieve most relevant chunks for a query using cosine similarity"""
    query_embedding = embedding_model.encode(
        query, 
        convert_to_tensor=True
    )
    
    # Calculate cosine similarity
    similarities = cos_sim(query_embedding, embeddings)[0]
    
    # Get top k indices
    top_k_indices = torch.topk(similarities, k=min(k, len(chunks))).indices
    
    # Get relevant chunks and their scores
    relevant_chunks = [chunks[i] for i in top_k_indices]
    scores = [similarities[i].item() for i in top_k_indices]
    
    return relevant_chunks, scores

def generate_answer(query, context, api_key):
    """Generate answer using Gemini API"""
    genai.configure(api_key=api_key)
    
    # First, try to get available models
    try:
        available_models = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        
        if available_models:
            # Use the first available model
            model_name = available_models[0]
            model = genai.GenerativeModel(model_name)
        else:
            raise Exception("No models available for content generation")
    except:
        # Fallback to trying specific model names
        model_names = [
            'gemini-1.5-flash-latest',
            'gemini-1.5-pro-latest',
            'gemini-pro',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro',
            'models/gemini-pro'
        ]
        
        model = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                # Test if it works
                test_response = model.generate_content("Hello")
                break
            except:
                continue
        
        if model is None:
            raise Exception("Could not find a working model. Please click 'Test API & List Models' to see available models.")
    
    prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: {query}

Answer:"""
    
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("🤖 RAG System with Gemini API")
st.markdown("Upload documents and ask questions using Google's Gemini model")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("Gemini API Key", type="password", help="Get your API key from https://makersuite.google.com/app/apikey")
    
    if api_key and st.button("Test API & List Models"):
        try:
            genai.configure(api_key=api_key)
            models = genai.list_models()
            st.success("✅ API Key Valid!")
            st.write("Available models:")
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    st.code(model.name)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.header("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )
    
    chunk_size = st.slider("Chunk Size", 100, 1000, 500)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50)
    top_k = st.slider("Number of relevant chunks", 1, 10, 3)
    
    if st.button("Process Documents", type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one document")
        else:
            all_text = ""
            
            # Extract text from all files
            for file in uploaded_files:
                if file.name.endswith('.pdf'):
                    all_text += extract_text_from_pdf(file) + "\n\n"
                elif file.name.endswith('.txt'):
                    all_text += extract_text_from_txt(file) + "\n\n"
            
            # Chunk the text
            st.session_state.chunks = chunk_text(all_text, chunk_size, chunk_overlap)
            
            # Load embedding model and create embeddings
            embedding_model = load_embedding_model()
            st.session_state.embeddings = create_embeddings(
                st.session_state.chunks,
                embedding_model
            )
            
            st.success(f"✅ Processed {len(uploaded_files)} file(s) into {len(st.session_state.chunks)} chunks")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask Questions")
    
    if st.session_state.embeddings is None:
        st.info("👈 Please upload and process documents first")
    else:
        query = st.text_input("Enter your question:", placeholder="What is this document about?")
        
        if st.button("Get Answer", type="primary"):
            if not api_key:
                st.error("Please enter your Gemini API key in the sidebar")
            elif not query:
                st.error("Please enter a question")
            else:
                with st.spinner("Searching and generating answer..."):
                    try:
                        # Retrieve relevant chunks
                        embedding_model = load_embedding_model()
                        relevant_chunks, scores = retrieve_relevant_chunks(
                            query,
                            st.session_state.embeddings,
                            st.session_state.chunks,
                            embedding_model,
                            k=top_k
                        )
                        
                        # Create context from relevant chunks
                        context = "\n\n".join(relevant_chunks)
                        
                        # Generate answer
                        answer = generate_answer(query, context, api_key)
                        
                        # Display results
                        st.markdown("### 📝 Answer")
                        st.markdown(answer)
                        
                        with st.expander("📚 View Retrieved Context"):
                            for i, (chunk, score) in enumerate(zip(relevant_chunks, scores)):
                                st.markdown(f"**Chunk {i+1}** (Similarity Score: {score:.4f})")
                                st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                                st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

with col2:
    st.header("📊 Stats")
    if st.session_state.chunks:
        st.metric("Total Chunks", len(st.session_state.chunks))
        avg_chunk_length = sum([len(chunk) for chunk in st.session_state.chunks]) / len(st.session_state.chunks)
        st.metric("Avg Chunk Length", f"{avg_chunk_length:.0f} chars")
    else:
        st.info("No documents processed yet")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🎈 | Powered by Gemini 🤖")