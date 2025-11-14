# Deploying RAG Application to Streamlit Community Cloud

This document summarizes the steps and tips to deploy the RAG Streamlit app (`app.py`) to Streamlit Community Cloud.

## 1) Prepare repo
- Ensure `app.py` is at the repository root (it is).
- Commit and push all changes to your GitHub repository.

## 2) Requirements
- Streamlit Community Cloud installs dependencies from `requirements.txt`. We added a starter file.
- If the app fails with dependency errors, run locally in the same environment and run `pip freeze > requirements.txt` and push the resulting file.

## 3) Secrets (required)
- In the Streamlit app dashboard, go to the app settings → "Secrets" and add:

```
OPENAI_API_KEY = "sk-..."
```

- The app prefers `st.secrets["OPENAI_API_KEY"]` and will fallback to an environment variable (for local testing).

## 4) Persistence & Vector DB notes
- The app uses a local `./chroma_db` directory by default. Streamlit’s filesystem is ephemeral across redeploys; do not rely on it for permanent storage.
- Options:
  - Use a managed vector DB service (Pinecone, Weaviate, Chroma Cloud) for production.
  - Accept ephemeral storage and rebuild vectors when users upload documents (fine for demos).
  - Use an object store (S3) to persist uploaded files and sync DB files (advanced).

## 5) File uploads & limits
- The file uploader stores files in memory or temporary disk of the running instance. Keep uploaded PDFs reasonably small for demo apps.

## 6) Running locally
- Export your key manually or add it to a `.env` file in the repo root:

```
# .env
OPENAI_API_KEY=sk-...
```

- Run locally:

```bash
export OPENAI_API_KEY="sk-..."   # or rely on .env loaded by python-dotenv
streamlit run app.py
```

## 7) Debugging on Streamlit Cloud
- Check the app logs from the Streamlit Cloud dashboard for dependency errors or runtime exceptions.
- Common issues:
  - Missing package or wrong package versions in `requirements.txt`.
  - Out-of-memory when embedding large PDFs — reduce file sizes or embed server-side.
  - OpenAI rate limits or invalid API key.

## 8) Suggested next improvements
- Switch to a hosted vector DB for persistence and scalability.
- Add a maximum file size and progress indicators during embedding.
- Add usage/cost controls (limit number of tokens or queries per session).

## 9) Contact
If you'd like, I can:
- Update `requirements.txt` further (pin exact working versions),
- Add a `.streamlit/config.toml` if you need custom server settings,
- Implement an S3-backed persistence option for `chroma_db`.

