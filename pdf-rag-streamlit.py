# minimal_streamlit_rag.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

st.title("PDF Q&A")

# Load everything once
@st.cache_resource
def setup_rag():
    loader = PyPDFLoader("./data/MAResume.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model="llama3.2")
    return retriever, llm

retriever, llm = setup_rag()

# Question and answer
question = st.text_input("Question:")
if question:
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    answer = llm.invoke(prompt).content
    st.write("**Answer:**", answer)