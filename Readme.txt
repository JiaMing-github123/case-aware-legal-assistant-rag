# CASE-AWARE LEGAL ASSISTANT USING RAG

An AI-powered legal assistant designed to answer questions related to the Industrial Relations Act 1967 (Act 177). The system uses a Retrieval-Augmented Generation (RAG) architecture with FastAPI as the backend and an HTML frontend.

---

## System Requirements

### Operating System
- Windows 10/11 (Recommended)
- Linux or macOS (Should also work)

---

## Required Software

| Software | Recommended Version | Download |
|----------|---------------------|----------|
| Python | 3.10 or above | https://www.python.org/downloads/ |
| Node.js (includes npm) | v18 or above | https://nodejs.org/ |
| Git (Optional) | Latest | https://git-scm.com/downloads |

---

## Clone the Repository

Clone the project from GitHub:

```bash
git clone https://github.com/JiaMing-github123/case-aware-legal-assistant-rag.git
```

Navigate to the project directory:

```bash
cd case-aware-legal-assistant-rag
```

## Python Libraries

Install all required dependencies:

```bash
pip install -r requirements.txt
```

If you encounter dependency issues, the following packages can also be installed manually:

```bash
pip install fastapi
pip install uvicorn
pip install python-dotenv
pip install langchain
pip install langchain-community
pip install langchain-core
pip install langchain-huggingface
pip install langchain-groq
pip install chromadb
pip install sentence-transformers
pip install pypdf
pip install docx2txt
pip install python-multipart
pip install pydantic
pip install transformers
pip install torch
```

The backend server is implemented using **FastAPI**.

---

## Project Structure

```
Project Folder
│
├── api_server.py
├── .env
├── data/
│   └── act_177.pdf
├── Cases/
│   └── Docx/
├── ui/
│   └── legal-assistant.html
└── README.md
```

---

## Environment Variables

Create a `.env` file in the project root.

Example:

```text
GROQ_API_KEY=your_groq_api_key
```

Obtain a Groq API key from:

https://console.groq.com/keys

The application requires the `GROQ_API_KEY` environment variable before starting.

---

## Running the Backend Server

Open a terminal in the project directory and execute:

```bash
uvicorn api_server:app --reload
```

The server will start at:

```
http://127.0.0.1:8000
```

---

## Accessing the Application

Open your browser and visit:

```
http://127.0.0.1:8000
```

The frontend page will be served automatically by FastAPI.

---

## Exposing the Local Server (Optional)

If you want to make the application accessible over the Internet, use either of the following tools.

### Option 1 – LocalTunnel

Install (if necessary):

```bash
npm install -g localtunnel
```

Run:

```bash
npx localtunnel --port 8000
```

Official website:

https://theboroer.github.io/localtunnel-www/

---

### Option 2 – Untun

Run:

```bash
npx untun tunnel http://localhost:8000
```

Official website:

https://github.com/unjs/untun

---

## Notes

- Ensure all datasets are placed in the correct folders before running the application.
- Ensure the `.env` file contains a valid `GROQ_API_KEY`.
- The default backend port is **8000**.
- Internet access is required for the Groq LLM API.
- The Chroma vector database will be generated automatically during the first startup.

---

## Technologies Used

- Python
- FastAPI
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Sentence Transformers
- Groq LLM API
- SQLite
- HTML/CSS/JavaScript

---

## Author

Final Year Project
