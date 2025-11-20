# ResumeVoiceRag 🎤📄

A voice-enabled RAG (Retrieval-Augmented Generation) system that allows you to query multiple PDF resumes using natural language and receive spoken responses powered by AI.

## Overview

ResumeVoiceRag combines document retrieval, large language models, and text-to-speech technology to create an interactive resume screening assistant. Ask questions about candidates' experience, skills, and background, and get intelligent answers delivered both as text and audio.

## Features

- **Multi-Resume Processing**: Load and analyze multiple PDF resumes simultaneously
- **Intelligent Retrieval**: Uses vector embeddings to find relevant information across all resumes
- **Natural Language Queries**: Ask questions in plain English about candidate experience and qualifications
- **AI-Powered Answers**: Leverages Llama 3.2 to generate contextual responses
- **Voice Output**: Text-to-speech integration using ElevenLabs API for spoken responses
- **Local LLM**: Runs on Ollama for privacy and cost-effectiveness

## Technology Stack

- **LangChain**: Document processing and RAG pipeline orchestration
- **Ollama**: Local LLM inference (Llama 3.2) and embeddings (nomic-embed-text)
- **Chroma**: Vector database for semantic search
- **ElevenLabs API**: High-quality text-to-speech conversion
- **Pygame**: Audio playback

## Prerequisites

- Python 3.8+
- Ollama installed locally with models:
  - `llama3.2`
  - `nomic-embed-text`
- ElevenLabs API key (free tier available)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MasonAnderson4/ResumeVoiceRag.git
cd ResumeVoiceRag
```

2. Install dependencies:
```bash
pip install langchain langchain-community langchain-text-splitters langchain-ollama chromadb pygame requests
```

3. Install Ollama and pull required models:
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2
ollama pull nomic-embed-text
```

4. Set up your ElevenLabs API key:
   - Sign up at [ElevenLabs](https://elevenlabs.io/)
   - Replace the `ELEVENLABS_API_KEY` in the script with your key

## Usage

1. Place your PDF resumes in the `./data/` directory

2. Update the `pdf_files` list in the script with your resume filenames:
```python
pdf_files = ["./data/resume1.pdf", "./data/resume2.pdf"]
```

3. Run the script:
```bash
python final-rag-voice.py
```

4. Modify the query in the script or create your own:
```python
answer = ask("What programming languages do these candidates know?", speak=True)
```

### Example Queries

- "Has any of these people worked in a restaurant?"
- "Which candidates have experience with Python?"
- "What are the education backgrounds of these applicants?"
- "Who has project management experience?"
- "Which candidate has the most years of experience?"

## How It Works

1. **Document Loading**: PDF resumes are loaded and converted to text
2. **Text Chunking**: Documents are split into manageable chunks (1000 characters)
3. **Embedding**: Chunks are converted to vector embeddings using nomic-embed-text
4. **Vector Storage**: Embeddings are stored in Chroma for efficient retrieval
5. **Query Processing**: User questions are embedded and matched against stored vectors
6. **Context Retrieval**: Relevant document chunks are retrieved
7. **Answer Generation**: Llama 3.2 generates an answer based on retrieved context
8. **Voice Synthesis**: ElevenLabs converts the answer to natural-sounding speech
9. **Audio Playback**: Response is played through speakers

## Configuration

### Disable Voice Output
```python
answer = ask("Your question here", speak=False)
```

### Change Voice
Update the `VOICE_ID` variable with a different ElevenLabs voice ID:
```python
VOICE_ID = "your_voice_id_here"
```

### Adjust Chunk Size
Modify the text splitter configuration:
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
```

### Change LLM Model
Update to use a different Ollama model:
```python
llm = ChatOllama(model="llama3.1")
```

## API Key Security

**Important**: Never commit your API keys to version control. Consider using environment variables:

```python
import os
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
```

## Limitations

- Requires active internet connection for ElevenLabs API
- PDF parsing quality depends on document formatting
- Response quality depends on LLM model capabilities
- ElevenLabs free tier has character limits

## Future Enhancements

- [ ] Add speech-to-text for voice queries
- [ ] Implement conversation memory for follow-up questions
- [ ] Add support for other document formats (Word, HTML)
- [ ] Create web interface with Streamlit or Gradio
- [ ] Add candidate comparison features
- [ ] Implement resume ranking based on job requirements

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- LangChain for RAG framework
- Ollama for local LLM inference
- ElevenLabs for text-to-speech API
- Chroma for vector database
