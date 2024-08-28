# Groq with Llama3 Assistant 🤖

This project is a Streamlit-based web application that leverages the power of Groq's Llama3 model and HuggingFace embeddings to create an intelligent document assistant. Users can upload PDFs, ask questions, and receive accurate answers based on the content of their documents.

## 🚀 Features

- 📚 **PDF Upload**: Easily upload and process multiple PDF documents.
- 🧩 **Document Vectorization**: Automatically create a vectorized document database for efficient information retrieval.
- ❓ **Question Answering**: Ask questions based on the content of your uploaded PDFs.
- 🎯 **Context-Aware Responses**: Receive precise answers backed by relevant document context.
- 🔍 **Document Similarity Search**: View similar sections of your documents related to your query.
- 📖 **Pre-loaded Classic Tales**: Includes support for questions about "The Kabuliwala", and "The Postmaster" by Rabindranath Tagore.

## 🛠️ Technologies Used

- Streamlit
- Langchain
- Groq API (Llama3 model)
- HuggingFace Embeddings
- FAISS for vector storage
- PyPDF for PDF processing

## 📋 Prerequisites

- Python 3.7+
- Groq API key

## 🔧 Installation

1. Clone this repository:
   ```
   git clone https://github.com/Aishee06/groq-llama3-assistant.git
   cd groq-llama3-assistant
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Create a directory named `pdfs` in the project root and place your PDF files there.

## 🚀 Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Click on "Create Vector Store from PDFs" to process and vectorize your documents.

4. Enter your question in the text input field and press Enter to get an answer.

5. Explore the Document Similarity Search results to see related content from your PDFs.