# README for RAG Application Deployment

## Overview
This is a Retrieval Augmented Generation (RAG) application built with LangChain, OpenAI, and Streamlit. It allows users to upload PDF documents and ask questions about their content.

## Features
- 📄 PDF document upload and processing
- 🔍 Semantic search using ChromaDB vector store
- 🤖 GPT-3.5-turbo powered responses
- 🎨 Clean and intuitive Streamlit UI
- ⚙️ Configurable model parameters (temperature, number of retrieved chunks)
- 📊 Document chunking with overlap for better context

## Local Setup

### Prerequisites
- Python 3.9 or higher
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-tutorial
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Click "New app"
4. Select your GitHub repository, branch, and `app.py` file
5. Click "Deploy"
6. Add your `OPENAI_API_KEY` in the Streamlit Cloud secrets manager (Settings → Secrets)

### Hugging Face Spaces (Docker)

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space)
2. Select "Docker" as the SDK
3. Push the following files to your Space repository:
   - `app.py` - Main Streamlit application
   - `requirements.txt` - Python dependencies
   - `Dockerfile` - Docker configuration
   - `.env` - Environment variables (add via Space settings/secrets)

4. The Space will automatically build and deploy

### Using Secrets on Hugging Face Spaces

For security, add your OpenAI API key as a secret:
1. Go to your Space settings
2. Click "Repository secrets"
3. Add `OPENAI_API_KEY` with your OpenAI API key value

## Project Structure

```
rag-tutorial/
├── app.py                          # Main Streamlit application
├── rag-notebook.ipynb              # Jupyter notebook with RAG implementation
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── README.md                       # This file
├── .env                           # Environment variables (not in git)
├── .env.example                   # Example environment variables
└── chroma_db/                     # Vector store persistence directory
```

## How It Works

### Three-Step RAG Process:

1. **Retrieval**: The app searches the ChromaDB vector store to find the most relevant document chunks based on the user's query.

2. **Augmentation**: The retrieved chunks are added to the prompt as context.

3. **Generation**: The GPT-3.5-turbo model uses this context to generate an accurate, grounded answer.

## Configuration

### Model Parameters (adjustable in sidebar)
- **Temperature** (0.0 - 1.0): Controls creativity vs. determinism
  - Lower values: More focused, deterministic responses
  - Higher values: More creative, varied responses

- **Number of Retrieved Chunks (k)** (1 - 10): How many document chunks to retrieve for context
  - Lower values: Faster but potentially less context
  - Higher values: More context but may dilute focus

## Dependencies

- `streamlit` - Web app framework
- `langchain` - LLM framework
- `langchain-openai` - OpenAI integration
- `langchain-community` - Community integrations (PDF loader, Chroma)
- `chromadb` - Vector database
- `openai` - OpenAI API client
- `pypdf` - PDF processing
- `python-dotenv` - Environment variable management

## Troubleshooting

### "ModuleNotFoundError: No module named 'langchain'"
- Ensure you've installed requirements: `pip install -r requirements.txt`
- Check your virtual environment is activated

### "OpenAI API key not found"
- Create a `.env` file in the project root with: `OPENAI_API_KEY=your_key`
- On Hugging Face Spaces, add it as a secret in space settings

### Document not processing
- Ensure the PDF is not corrupted
- Check file size (very large PDFs may cause issues)
- Verify OpenAI API key is valid

### Slow responses
- Reduce the number of retrieved chunks (k parameter)
- This speeds up processing at the cost of less context

## Future Enhancements

- [ ] Support for multiple file formats (DOCX, TXT, etc.)
- [ ] Batch document processing
- [ ] Chat history and session management
- [ ] Different LLM model options
- [ ] Advanced filtering and keyword search
- [ ] Response confidence scoring

## License
MIT License

## Support
For issues or questions, please create an issue in the GitHub repository.
