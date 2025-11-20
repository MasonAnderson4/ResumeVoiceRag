# minimal_rag.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Load PDF
loader = PyPDFLoader("./data/MAResume.pdf")
docs = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_documents(docs)

# Create vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings)

# Setup retriever and LLM
retriever = vectorstore.as_retriever()
llm = ChatOllama(model="llama3.2")

# Function to ask questions
def ask(question):
    # Get relevant chunks
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    
    # Create prompt
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    # Get answer
    return llm.invoke(prompt).content

# Test it
answer = ask("What experience does this person have?")
print(answer)