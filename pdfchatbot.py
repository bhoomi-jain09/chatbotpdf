import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os

load_dotenv()

st.title("📄 RAG-based PDF Chatbot")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Upload
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None and st.session_state.vectorstore is None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        file_bytes = uploaded_file.read()
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(chunks, embedding_model)


    st.session_state.vectorstore = vectorstore

    st.success("PDF processed successfully!")


    os.remove(tmp_path)

# Chat UI 
if st.session_state.vectorstore is not None:
    query = st.chat_input("Ask a question about the PDF:")

    if query:
        with st.chat_message("user"):
            st.write(query)
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

        retrieved_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        model = ChatGroq(model="llama-3.1-8b-instant")

        prompt = ChatPromptTemplate.from_messages([
            ("system",
            """You are a helpful AI assistant.
            Use ONLY the provided context to answer.

        Rules:
        - If answer is not in context → say "I don't know based on the document."
        - Keep answer clear and easy to understand
        - Add a short explanation or example if helpful
        - Keep tone friendly and human-like

        Context:{context}
        """),
            ("human", "{input}")
        ])

        chain = prompt | model

        response = chain.invoke({
            "context": context,
            "input": query
        })

        st.markdown("### 🤖 Answer:")
        st.write(response.content)

else:
    st.info("👆 Please upload a PDF to start chatting.")