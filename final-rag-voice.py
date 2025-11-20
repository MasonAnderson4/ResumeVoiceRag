# rag_with_voice_fixed.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
import requests
import pygame
import tempfile
import os

# ElevenLabs Configuration
ELEVENLABS_API_KEY = "sk_725d7d05cd76e77d469cd5fc159e4925bf187c15b667fd99"
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice

# Load PDF
pdf_files = ["./data/3547447.pdf", "./data/10001727.pdf", "./data/10030015.pdf", 
             "./data/10235429.pdf", "./data/10554236.pdf"]
all_docs = []
for pdf_file in pdf_files:

    loader = PyPDFLoader(pdf_file)
    all_docs.extend(loader.load())

# Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_documents(all_docs)

# Create vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings)

# Setup retriever and LLM
retriever = vectorstore.as_retriever()
llm = ChatOllama(model="llama3.2")

def text_to_speech(text):
    """Convert text to speech using ElevenLabs API with updated model"""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    data = {
        "text": text,
        "model_id": "eleven_turbo_v2",  # Updated model for free tier
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.content
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def play_audio(audio_data):
    """Play audio using pygame"""
    pygame.mixer.init()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        temp_file.write(audio_data)
        temp_file_path = temp_file.name
    
    pygame.mixer.music.load(temp_file_path)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    os.remove(temp_file_path)

def ask(question, speak=True):
    # Get relevant chunks
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    
    # Create prompt
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    # Get answer
    answer = llm.invoke(prompt).content
    
    # Print the answer
    print(f"\n📝 Answer: {answer}")
    
    # Convert to speech and play if requested
    if speak:
        print("🔊 Generating audio...")
        audio_data = text_to_speech(answer)
        if audio_data:
            print("▶️ Playing audio...")
            play_audio(audio_data)
        else:
            print("❌ Failed to generate audio")
    
    return answer

# Test it
answer = ask("Has any of these people worked in a resturant?", speak=True)