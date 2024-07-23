from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.pdf import PyPDFLoader
import os
import numpy as np
import time
import math
import random
import pprint
import nltk
from os.path import isfile, join

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st

dataset_path = "/dataset/"
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}
# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

def load_embeddings(modelPath, model_kwargs, encode_kwargs):
    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )


    print("\n----------------Test document Embeddings--------------------")
    text = "This is a test document."
    query_result = embeddings.embed_query(text)
    #print(query_result[:3])
    
    return embeddings
    
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""
if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question")
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = FAISS(persist_directory='jj',
                                          embedding_function=load_embeddings(modelPath, model_kwargs, encode_kwargs)
                                          )
if 'llm' not in st.session_state:
        # Specify the model name you want to use
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    # Load the tokenizer associated with the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)
    
    # Define a question-answering pipeline using the model and tokenizer
    question_answerer = pipeline(
        "text2text-generation", 
        model=model_name, 
        tokenizer=tokenizer,
        max_new_tokens=512,
        model_kwargs = {'device_map':"auto"}
        #return_tensors='pt'
    )
    
    # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
    # with additional model-specific arguments (temperature and max_length)
    llm = HuggingFacePipeline(
        pipeline=question_answerer,
        model_kwargs=model_kwargs,
    )
    st.session_state.llm = llm

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("PDF Chatbot")
# Upload a PDF file
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

def read_files(dataset_path, sflag):    
    onlyfiles = [f for f in os.listdir(dataset_path) if isfile(join(dataset_path, f))]
    # Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
    # It splits text into chunks of 1000 characters each with a 150-character overlap.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    data = []
    docs=[]

    for file in onlyfiles:
        if file.endswith('.pdf'):
            loader = PyPDFLoader(dataset_path+file)
            p = loader.load_and_split(text_splitter)
            if sflag==True:
                random.shuffle(p)
            for page in p:
                docs.append(page)
    return docs


    
if uploaded_file is not None:
    if not os.path.isfile("files/"+uploaded_file.name+".pdf"):
        with st.status("Analyzing your document..."):
            docs = read_files(dataset_path, False)
            # Create and persist the vector store
            embeddings = load_embeddings(modelPath, model_kwargs, encode_kwargs)
            st.session_state.vectorstore = FAISS.from_documents(
                documents = docs, 
                embeddings = embeddings)
            st.session_state.vectorstore.persist()
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    # Initialize the QA chain
    
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

        # Chat input
    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response['result'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        chatbot_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(chatbot_message)


else:
    st.write("Please upload a PDF file.")
