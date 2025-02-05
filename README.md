# DocuMind AI

## Overview
DocuMind AI is an intelligent document assistant that leverages advanced AI models to analyze and retrieve information from uploaded documents. Built with Streamlit, LangChain, Ollama, and Groq, it provides an interactive experience for querying documents using a RAG (Retrieval-Augmented Generation) system.

## Features
- **Retrieval-Augmented Generation (RAG) System** for document-based Q&A
- **Multi-document support** for PDFs, TXT, and DOCX files
- **Summarization & search** capabilities using embeddings
- **Customizable AI models** from Ollama and Groq
- **Interactive chat interface** with a modern UI

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python, FastAPI (Future Scope)
- **AI Models:** Ollama, Groq
- **Embeddings:** OllamaEmbeddings (nomic-embed-text)
- **Vector Database:** InMemoryVectorStore (LangChain)
- **Document Loaders:** PDFPlumberLoader, UnstructuredFileLoader

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/DocuMind-AI.git
   cd DocuMind-AI
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add your API key:
   ```bash
   echo "GROQ_API_KEY=your_api_key_here" > .env
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Upload one or more documents (.pdf, .txt, .docx).
3. Enter your query in the chat input.
4. Get AI-powered responses based on the document content.

## Configuration
The sidebar allows you to:
- Choose between **Ollama** and **Groq** as your AI provider.
- Select different model variants like `mixtral-8x7b-32768`, `llama3-70b-8192`, etc.

## Future Enhancements
- **Django-based backend** for scalability
- **Database integration** for persistent storage
- **Fine-tuned models** for domain-specific analysis
- **Web-based deployment** using Docker

## License
MIT License

## Contributors
- Nishant Chopra

For issues and feature requests, please open an issue on [GitHub](https://github.com/your-repo/DocuMind-AI/issues).

