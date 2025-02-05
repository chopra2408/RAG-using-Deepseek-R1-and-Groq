import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredFileLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.markdown(
    """
    <style>
    .stApp{
        background-color: #0E1117;
        color: #FFFFFF;
    }        
    
    .stChatInput input{
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd){
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even){
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stChatMessage .avatar{
        background-color: #00FFAA !important;
        color: #FFFFFF !important;
    }
    
    .stChatMessage p, .stChatMessage div{
        color: #FFFFFF !important;
    }
    
    .stFileUploader{
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3{
        color: #00FFAA !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

prompt_template = """
You are an expert research assistant. Use the provided context to answer the query.
If unsure, state that you don't know. Be concise and factual.

Query: {user_query}
Context: {document_context}
Answer:
"""

docstore_path = "./docstore/"
os.makedirs(docstore_path, exist_ok=True)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectordb = InMemoryVectorStore(embeddings)

def save_uploaded_files(uploaded_files):
    saved_paths = []
    for file in uploaded_files:
        file_path = os.path.join(docstore_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PDFPlumberLoader(file_path)
        return loader.load()
    elif ext in [".txt"]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        from langchain.schema import Document
        return [Document(page_content=text)]
    elif ext in [".docx"]:
        loader = UnstructuredFileLoader(file_path)
        return loader.load()
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        from langchain.schema import Document
        return [Document(page_content=text)]

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100, 
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    vectordb.add_documents(document_chunks)

def find_related_documents(query):
    return vectordb.similarity_search(query)

def generate_answer(user_query, context_documents, llm_instance):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = conversation_prompt | llm_instance
    return chain.invoke({"user_query": user_query, "document_context": context_text})

st.title("DocuMind AI")
st.markdown("### Your intelligent assistant")
st.markdown("---")

with st.sidebar:
    st.header("Configuration")
    
    # Model provider selection
    provider = st.selectbox(
        "Choose Provider",
        ["Ollama", "Groq"],
        index=0,
        help="Select AI provider"
    )
    
    # Model selection based on provider
    if provider == "Ollama":
        model_options = ["deepseek-r1:1.5b", "llama3.2:3b"]
    else:
        model_options = ["mixtral-8x7b-32768", "llama3-70b-8192", "gemma2-9b-it"]
    
    selected_model = st.selectbox(
        "Choose Model",
        model_options,
        index=0,
        help="Select model variant"
    )
    
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
                - RAG System
                - Summarizes Documents
                - Multi Document Support
                - Easy to use
            """)
    
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [Groq](https://groq.com/) | [Langchain](https://python.langchain.com/)")

# Initialize appropriate LLM engine based on the provider.
if provider == "Ollama":
    llm_engine = OllamaLLM(model=selected_model)
else:
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables!")
        st.stop()
    llm_engine = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=selected_model,
        temperature=0.1
    )
    
upload_files = st.file_uploader(
    "Upload Documents", 
    type=["pdf", "txt", "docx"], 
    help="Select one or more documents for analysis",
    accept_multiple_files=True
)

if upload_files:
    saved_paths = save_uploaded_files(upload_files)
    all_chunks = []
    for path in saved_paths:
        raw_documents = load_document(path)
        processed_chunks = chunk_documents(raw_documents)
        all_chunks.extend(processed_chunks)
    
    index_documents(all_chunks)
    st.success(f"Processed {len(upload_files)} document(s) successfully!")
    
    st.sidebar.subheader("Uploaded Documents")
    for path in saved_paths:
        st.sidebar.markdown(f"- `{os.path.basename(path)}`")
    
    if st.sidebar.button("Clear Documents"):
        vectordb.clear()
        st.session_state.clear()
        st.experimental_rerun()

    # Main area: Chat input for user question.
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
            
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs, llm_engine)
            
        with st.chat_message("assistant", avatar="😁"):
            st.write(ai_response)
