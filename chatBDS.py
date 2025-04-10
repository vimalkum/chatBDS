import os
import time
import random
import glob
import streamlit as st
from transformers import AutoTokenizer, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.badges import badge

# Set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Paths
DATASET_PATH = "/home/vkumar/code/dataset/BDS_data/"
MODEL_PATH = "/home/vkumar/code/Models/bge-large-en-v1.5"
MODEL_NAME = '/home/vkumar/code/Models/Marcoro14-7B-slerp/'
FAISS_INDEX_PATH = '/home/vkumar/code/faiss_index/'

# Model parameters
MODEL_KWARGS = {'device': 'cuda'}
ENCODE_KWARGS = {'normalize_embeddings': False}

def load_embeddings(model_path, model_kwargs, encode_kwargs):
    """Load embeddings from HuggingFace."""
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def read_files(dataset_path, shuffle=False):
    """Read and split PDF files into text chunks."""
    files = glob.glob(dataset_path + '/**/*.pdf', recursive=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    
    for file in files:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split(text_splitter)
        if shuffle:
            random.shuffle(pages)
        docs.extend(pages)
    
    return docs

def initialize_llm(model_name):
    """Initialize the language model pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    
    qa_pipeline = pipeline(
        "text2text-generation", 
        model=model_name, 
        tokenizer=tokenizer,
        max_new_tokens=512,
        model_kwargs={'device_map': 'auto'}
    )
    
    return HuggingFacePipeline(pipeline=qa_pipeline, model_kwargs=MODEL_KWARGS)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'template' not in st.session_state:
        st.session_state.template = """
        You are a knowledgeable chatbot, here to help with user questions. Your tone should be professional and informative.
        
        Context: {context}
        History: {history}
        
        User: {question}
        Chatbot:"""
    
    if 'prompt' not in st.session_state:
        st.session_state.prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=st.session_state.template
        )
    
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            input_key="question"
        )
    
    if 'llm' not in st.session_state:
        st.session_state.llm = initialize_llm(MODEL_NAME)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def setup_vectorstore():
    """Setup FAISS vector store."""
    embeddings = load_embeddings(MODEL_PATH, MODEL_KWARGS, ENCODE_KWARGS)
    if os.path.exists(FAISS_INDEX_PATH):
        st.session_state.vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = read_files(DATASET_PATH, shuffle=False)
        st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
        st.session_state.vectorstore.save_local(FAISS_INDEX_PATH)
    
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory
            }
        )

# Streamlit UI
st.title("🌟 BDS Chatbot 🌟")
st.markdown("<h3 style='color:blue;'>Your AI-powered assistant for document analysis</h3>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image('./logo-bds-black.png')
    st.title('🤗💬 ChatBDS App')
    st.markdown('''
    ## 🌟 About
    This app is a LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain]
    - [LLM - Marcoro14-7B]
    ''')
    add_vertical_space(2)
    st.markdown("<h4 style='color:green;'>Made with ❤️ by [ChatBDS Team]</h4>", unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

if 'initial' not in st.session_state:
    st.session_state.initial = True

message_placeholder = st.empty()

if st.session_state.initial:
    if os.path.isdir(DATASET_PATH):
        with st.status("🔍 Analyzing your document..."):
            setup_vectorstore()
    st.session_state.initial = False

# Chat interaction
if user_input := st.chat_input("📝 You:"):
    user_message = {"role": "user", "message": user_input}
    st.session_state.chat_history.append(user_message)
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("🤖 Assistant is typing..."):
            response = st.session_state.qa_chain(user_input)
        
        full_response = ""
        for chunk in response.get('result', '').strip().split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        
        st.markdown(f"<div style='color:darkblue; font-weight:bold;'>{full_response}</div>", unsafe_allow_html=True)
    
    chatbot_message = {"role": "assistant", "message": response.get('result', '').strip()}
    st.session_state.chat_history.append(chatbot_message)
