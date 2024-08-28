import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Load the API key for Groq
groq_api_key = os.getenv('GROQ_API_KEY')

# Set page configuration
st.set_page_config(
    page_title="Groq with Llama3 Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Title and Description
st.title("🤖 Groq with Llama3 Assistant")
st.markdown(
    """
    ### 🚀 Turbocharge Your Workflow with Groq's Llama3 ⚡ and HuggingFace Magic ✨!
    **📄 Features**:
    - 📚 **Upload PDFs**: Easily upload and process your PDFs.
    - 🧩 **Vectorize Documents**: Automatically create a vectorized document database for efficient information retrieval.
    - ❓ **Ask Questions Based on Your Documents**: Enter your queries, and let the assistant search through your PDFs for the most accurate answers.
    - 🎯 **Get Precise Answers with Context**: Receive answers that are not just correct but are backed by relevant document context.
    - 🔗 **Document Similarity Search**: See similar sections of your documents that are related to your query.
    """
)

# Add the new lines about the books
st.markdown(
    """
    📚 Dive Into Classic Tales!  
    Got questions about **"The Kabuliwala"**, or **"The Postmaster"** by Rabindranath Tagore? 🎩🔎 Our assistant is here to help you unravel the magic of these timeless stories! Just type your question, and we'll find the answers you seek from these captivating reads. 🌟📖
    """
)

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def vector_embedding():
    with st.spinner("Building Vector Store..."):
        if "vectors" not in st.session_state:
            # Use HuggingFaceEmbeddings instead of OpenAIEmbeddings
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.loader = PyPDFDirectoryLoader("./pdfs")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector Store DB is Ready!")

# User Input
st.markdown("### 🧠 Curious? Ask Away!")
prompt1 = st.text_input("Enter Your Question", placeholder="Type your question here...")

# Embed Documents Button
if st.button("Create Vector Store from PDFs"):
    vector_embedding()

# Generate Answer
if prompt1:
    with st.spinner("Retrieving the best answer..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start:.2f} seconds")

        # Display Answer
        st.markdown("### 🎯 Your Answer")
        st.success(response['answer'])

        # Display Document Similarity Search in an Expander
        with st.expander("🔍 Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Document {i+1}**")
                st.write(doc.page_content)
                st.write("---")
