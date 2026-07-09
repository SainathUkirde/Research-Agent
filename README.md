# ResearchAI — AI-Powered Research Agent

> Built on **IBM watsonx.ai** · **IBM Granite** · **Flask** · **ChromaDB** · **arXiv** · **Semantic Scholar**

An end-to-end academic research assistant that helps you discover, understand, synthesise, and draft scientific literature — powered exclusively by IBM Granite for all generative tasks.

---

## Features

| Feature | Description |
|---|---|
| 🤖 **AI Chat** | Conversational Q&A grounded in your saved papers (RAG) |
| 🔍 **Literature Search** | Simultaneous search across arXiv and Semantic Scholar |
| 📝 **Paper Summarization** | Structured summaries (Problem / Approach / Findings / Significance) |
| 💡 **Hypothesis Lab** | Research gap identification and testable hypothesis generation |
| ✍️ **Draft Section** | Auto-draft Introduction, Related Work, Methodology, or Conclusion |
| 📚 **Reference Library** | Save, tag, and organize papers by project |
| 📎 **Citation Export** | APA, MLA, IEEE formatted citations and BibTeX export |
| 🌙 **Dark Mode** | Full dark/light theme toggle |

---

## Tech Stack

```
Backend   : Python 3.11+, Flask 3, Flask-SQLAlchemy, Flask-Cors
AI Engine : IBM watsonx.ai SDK (ibm-watsonx-ai), Granite-3-3-8b-instruct
RAG       : ChromaDB (swappable to FAISS), sentence-transformers (all-MiniLM-L6-v2)
APIs      : arXiv Python library, Semantic Scholar REST API
Database  : SQLite (via SQLAlchemy ORM)
Frontend  : Bootstrap 5, custom CSS theme, vanilla JavaScript
```

---

## Project Structure

```
research-agent/
├── app.py                  # Flask application + all REST endpoints
├── agent_config.py         # AGENT_INSTRUCTIONS — edit to customize Granite's persona
├── watsonx_client.py       # IBM watsonx.ai SDK wrapper (generate_text, generate_chat)
├── rag_pipeline.py         # Modular RAG pipeline (ChromaDB / FAISS, swappable)
├── paper_apis.py           # arXiv + Semantic Scholar + citation formatters
├── models.py               # SQLAlchemy ORM models (SavedPaper, ChatMessage)
├── requirements.txt
├── .env.example            # ← copy to .env and fill in credentials
├── DEPLOYMENT.md           # Full IBM Cloud deployment guide
├── templates/
│   ├── base.html           # Sidebar layout, dark mode toggle
│   ├── index.html          # Home / feature dashboard
│   ├── chat.html           # AI chat interface with source snippets
│   ├── dashboard.html      # Literature search + summarize
│   ├── library.html        # Reference manager + BibTeX export
│   ├── draft.html          # Paper section drafter
│   └── hypothesis.html     # Research hypothesis + gap analysis
└── static/
    ├── css/theme.css       # Custom theme, dark mode, animations
    └── js/app.js           # Dark mode, toast notifications, utilities
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/research-agent.git
cd research-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the embedding model (~90 MB) on first run. Subsequent starts use the local cache.

### 4. Configure credentials

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
WATSONX_API_KEY=your_ibm_cloud_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
FLASK_SECRET_KEY=any_long_random_string
```

See [Getting IBM watsonx.ai Credentials](#getting-ibm-watsonxai-credentials) below.

### 5. Run the app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

Verify the IBM Granite connection at **http://localhost:5000/api/health** — you should see:
```json
{"status": "ok", "sample": "OK"}
```

---

## Getting IBM watsonx.ai Credentials

### Step 1 — Create a free IBM Cloud account
Go to [https://cloud.ibm.com/registration](https://cloud.ibm.com/registration) and sign up for a **Lite** (free) account.

### Step 2 — Provision watsonx.ai
1. In the IBM Cloud catalog, search for **watsonx.ai**
2. Select the **Lite** plan → choose region `us-south` → click **Create**
3. Click **Launch watsonx.ai** to open the studio

### Step 3 — Create a project and get the Project ID
1. In the studio: **New project** → **Create an empty project**
2. Open the project → **Manage** tab → **General**
3. Copy the **Project ID** → set as `WATSONX_PROJECT_ID`

### Step 4 — Get your API key
1. IBM Cloud console → avatar (top-right) → **Profile and settings**
2. **IBM Cloud API keys** → **Create an IBM Cloud API key**
3. Copy the key immediately (shown once) → set as `WATSONX_API_KEY`

### Step 5 — Confirm Granite model availability
In watsonx.ai studio, verify that `ibm/granite-3-3-8b-instruct` is available in your region.  
If you see a different model ID, update `GRANITE_CONFIG["model_id"]` in [`agent_config.py`](agent_config.py).

---

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health` | Check IBM watsonx.ai connection |
| `POST` | `/api/search` | Search arXiv + Semantic Scholar |
| `POST` | `/api/chat` | RAG-grounded Q&A (session-aware) |
| `GET`  | `/api/chat/history` | Retrieve chat history for a session |
| `POST` | `/api/chat/clear` | Clear chat history |
| `POST` | `/api/summarize` | Summarize a paper abstract or text |
| `POST` | `/api/hypothesis` | Generate research gaps + hypotheses |
| `POST` | `/api/draft` | Draft a paper section with citations |
| `POST` | `/api/citations` | Format APA / MLA / IEEE citations |
| `GET`  | `/api/library` | List saved papers (filter by tag/project) |
| `POST` | `/api/library` | Save a paper + auto-ingest into RAG |
| `PATCH`| `/api/library/<id>` | Update paper tags, project, notes |
| `DELETE`| `/api/library/<id>` | Remove a paper from the library |
| `GET`  | `/api/library/export/bibtex` | Download library as `.bib` file |
| `POST` | `/api/ingest` | Manually ingest a paper into the vector store |

### Example: Chat request

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the key challenges in LLM alignment?"}'
```

### Example: Literature search

```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "federated learning privacy", "sources": ["arxiv", "semantic_scholar"], "max_results": 8}'
```

---

## Customizing the Agent

All agent persona and behaviour is controlled by `AGENT_INSTRUCTIONS` in [`agent_config.py`](agent_config.py):

```python
AGENT_INSTRUCTIONS = {
    "role": "You are a rigorous academic research assistant...",
    "tone": "formal, precise, evidence-based",   # conversational | concise | formal
    "domain_specialization": [],                  # e.g. ["machine learning", "genomics"]
    "citation_style": "APA",                      # APA | MLA | IEEE
    "safety_rules": [
        "Never fabricate citations, authors, or paper content.",
        "Always distinguish between retrieved evidence and model inference.",
        ...
    ],
    "output_preferences": {
        "include_confidence_notes": True,
        "max_summary_length_words": 200,
    }
}
```

Edit this block and restart the server — the new instructions are injected into every Granite prompt automatically via `build_system_prompt()`. No other code changes needed.

---

## Swapping the Vector Store

The RAG backend is fully modular. Change one key in `agent_config.py`:

```python
VECTOR_STORE_CONFIG = {
    "backend": "faiss",   # change from "chroma" to "faiss"
    ...
}
```

Install FAISS if switching:
```bash
pip install faiss-cpu
```

No changes to routes or `app.py` are required.

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `WATSONX_API_KEY` | IBM Cloud API key | ✅ |
| `WATSONX_PROJECT_ID` | watsonx.ai project UUID | ✅ |
| `WATSONX_URL` | Regional endpoint (e.g. `https://us-south.ml.cloud.ibm.com`) | ✅ |
| `FLASK_SECRET_KEY` | Flask session secret | ✅ |
| `FLASK_DEBUG` | `true` for development mode | ❌ |
| `PORT` | Port to bind (default: `5000`) | ❌ |
| `DATABASE_URL` | SQLAlchemy URL (default: `sqlite:///research_agent.db`) | ❌ |

---

## Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for complete step-by-step instructions including:

- IBM Code Engine (serverless, recommended)
- IBM Cloud Foundry
- Local Docker

---

## Security Notes

- Credentials are **only** loaded from `.env` via `python-dotenv` — never hardcoded
- `.env` is listed in `.gitignore` — it will not be committed
- The agent's safety rules explicitly forbid fabricating citations or presenting unverified claims

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for the IBM watsonx.ai Challenge — IBM Granite is used exclusively for all generative tasks.*
