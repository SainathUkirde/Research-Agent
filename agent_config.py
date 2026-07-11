"""
agent_config.py
---------------
Centralised, easy-to-edit configuration for the Research Agent persona,
behaviour, and output preferences.  Import AGENT_INSTRUCTIONS into any
module that builds prompts for Granite — never touch core logic to change
the agent's tone or rules.
"""

AGENT_INSTRUCTIONS = {
    # ── Identity ──────────────────────────────────────────────────────────
    "role": (
        "You are a rigorous academic research assistant powered by IBM Granite. "
        "You help researchers discover, understand, summarise, and synthesise "
        "scientific literature. You ground every claim in retrieved source material "
        "and clearly flag when you are inferring beyond what the sources say."
    ),
    # ── Communication style ───────────────────────────────────────────────
    "tone": "formal, precise, evidence-based",   # options: conversational | concise | formal
    # ── Domain focus (empty = general; add strings to narrow) ────────────
    "domain_specialization": [],                  # e.g. ["machine learning", "bioinformatics"]
    # ── Citation preferences ──────────────────────────────────────────────
    "citation_style": "APA",                      # APA | MLA | IEEE
    # ── Hard safety rules injected into every prompt ─────────────────────
    "safety_rules": [
        "Never fabricate citations, author names, paper titles, or DOIs.",
        "Always distinguish between retrieved evidence and model inference.",
        "Flag explicitly when a claim cannot be verified from the provided sources.",
        "Do not present unpublished / preprint findings as peer-reviewed consensus.",
        "If no relevant sources are available, say so rather than hallucinating.",
        "Avoid reproducing copyrighted text verbatim; paraphrase and cite instead.",
    ],
    # ── Output formatting preferences ─────────────────────────────────────
    "output_preferences": {
        "include_confidence_notes": True,    # append a brief confidence note after answers
        "max_summary_length_words": 200,     # target word count for /api/summarize
        "use_bullet_points": True,           # use bullet points for hypothesis & gap lists
        "include_inline_citations": True,    # embed [Author, Year] markers in drafted text
    },
}

# ── Granite model config ──────────────────────────────────────────────────────
GRANITE_CONFIG = {
    "model_id": "ibm/granite-4-h-small",            # latest supported Granite instruct model
    "parameters": {
        "decoding_method": "greedy",
        "max_new_tokens": 1024,
        "min_new_tokens": 1,
        "temperature": 0.3,          # low temp for factual/academic tasks
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    },
}

# ── Embedding model config ────────────────────────────────────────────────────
EMBEDDING_CONFIG = {
    # Used for RAG chunking & retrieval (local sentence-transformers model)
    "model_name": "all-MiniLM-L6-v2",
    "chunk_size": 512,        # tokens per chunk
    "chunk_overlap": 64,      # overlap between consecutive chunks
    "top_k_retrieval": 5,     # how many chunks to feed Granite as context
}

# ── Vector store config ───────────────────────────────────────────────────────
VECTOR_STORE_CONFIG = {
    "backend": "chroma",                    # "chroma" | "faiss"
    "chroma_persist_dir": "./data/chroma_db",
    "faiss_index_path": "./data/faiss.index",
    "collection_name": "research_papers",
}

# ── API config ────────────────────────────────────────────────────────────────
API_CONFIG = {
    "arxiv_max_results": 10,
    "semantic_scholar_max_results": 10,
    "semantic_scholar_base_url": "https://api.semanticscholar.org/graph/v1",
    "request_timeout_seconds": 15,
}

# ── System prompt builder ─────────────────────────────────────────────────────
def build_system_prompt(task_hint: str = "") -> str:
    """Return the full system prompt string for a Granite API call."""
    rules_block = "\n".join(f"  - {r}" for r in AGENT_INSTRUCTIONS["safety_rules"])
    domains = (
        ", ".join(AGENT_INSTRUCTIONS["domain_specialization"])
        if AGENT_INSTRUCTIONS["domain_specialization"]
        else "general academic research"
    )
    prefs = AGENT_INSTRUCTIONS["output_preferences"]
    confidence_note = (
        "Append a brief confidence note (e.g. 'Confidence: High — directly supported by sources')"
        if prefs["include_confidence_notes"]
        else ""
    )
    return (
        f"{AGENT_INSTRUCTIONS['role']}\n\n"
        f"Tone: {AGENT_INSTRUCTIONS['tone']}\n"
        f"Domain focus: {domains}\n"
        f"Citation style: {AGENT_INSTRUCTIONS['citation_style']}\n"
        f"Task: {task_hint}\n\n"
        f"Non-negotiable rules:\n{rules_block}\n\n"
        f"{confidence_note}"
    ).strip()
