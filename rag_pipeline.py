"""
rag_pipeline.py
---------------
Modular RAG (Retrieval-Augmented Generation) pipeline.
- Ingest paper abstracts / full text into a vector store
- Chunk and embed text (local sentence-transformers)
- Retrieve top-k relevant chunks for a query
- Swap the vector store backend (ChromaDB / FAISS) by changing
  VECTOR_STORE_CONFIG["backend"] in agent_config.py — no route changes needed.
"""

import logging
import hashlib
from typing import Optional

from agent_config import EMBEDDING_CONFIG, VECTOR_STORE_CONFIG

logger = logging.getLogger(__name__)

# ── Lazy imports ──────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed.  RAG embeddings disabled.")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed.")

try:
    import faiss
    import numpy as np
    import json, os, pickle
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed.")


# ── Embedding helper ──────────────────────────────────────────────────────────
_embedder: Optional["SentenceTransformer"] = None


def _get_embedder() -> "SentenceTransformer":
    global _embedder
    if _embedder is None:
        if not ST_AVAILABLE:
            raise RuntimeError("sentence-transformers is required for RAG.")
        logger.info("Loading embedding model: %s", EMBEDDING_CONFIG["model_name"])
        _embedder = SentenceTransformer(EMBEDDING_CONFIG["model_name"])
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return a list of embedding vectors for the given texts."""
    embedder = _get_embedder()
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()


# ── Text chunking ─────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """
    Split *text* into overlapping word-based chunks.
    Uses EMBEDDING_CONFIG["chunk_size"] and ["chunk_overlap"].
    """
    words = text.split()
    size = EMBEDDING_CONFIG["chunk_size"]
    overlap = EMBEDDING_CONFIG["chunk_overlap"]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks


def _paper_id(paper: dict) -> str:
    """Generate a stable ID for a paper dict based on title + arxiv_id / doi."""
    key = (paper.get("arxiv_id") or paper.get("doi") or paper.get("title", "")).strip()
    return hashlib.md5(key.encode()).hexdigest()


# ── ChromaDB backend ──────────────────────────────────────────────────────────
_chroma_client = None
_chroma_collection = None


def _get_chroma_collection():
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        if not CHROMA_AVAILABLE:
            raise RuntimeError("chromadb not installed.")
        persist_dir = VECTOR_STORE_CONFIG["chroma_persist_dir"]
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=VECTOR_STORE_CONFIG["collection_name"],
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection ready: %s", VECTOR_STORE_CONFIG["collection_name"])
    return _chroma_collection


def ingest_chroma(paper: dict) -> int:
    """Add a paper's chunks to ChromaDB.  Returns number of chunks added."""
    col = _get_chroma_collection()
    text = (paper.get("abstract") or "") + " " + (paper.get("full_text") or "")
    if not text.strip():
        return 0
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)
    paper_id = _paper_id(paper)
    ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
    meta = {
        "paper_id": paper_id,
        "title": paper.get("title", ""),
        "authors": ", ".join(paper.get("authors", [])),
        "year": str(paper.get("year", "")),
        "arxiv_id": paper.get("arxiv_id", ""),
        "doi": paper.get("doi", ""),
    }
    # ChromaDB requires metadata per document
    metas = [meta.copy() for _ in chunks]
    col.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metas)
    logger.info("Ingested %d chunks for paper: %s", len(chunks), paper.get("title", ""))
    return len(chunks)


def retrieve_chroma(query: str, top_k: Optional[int] = None) -> list[dict]:
    """Return top-k relevant chunks from ChromaDB for *query*."""
    col = _get_chroma_collection()
    k = top_k or EMBEDDING_CONFIG["top_k_retrieval"]
    q_emb = embed_texts([query])[0]
    results = col.query(query_embeddings=[q_emb], n_results=k)
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        chunks.append({"text": doc, "metadata": meta})
    return chunks


# ── FAISS backend ─────────────────────────────────────────────────────────────
_faiss_index = None
_faiss_metadata: list[dict] = []
_faiss_documents: list[str] = []
_FAISS_META_PATH = VECTOR_STORE_CONFIG.get("faiss_index_path", "./data/faiss.index") + ".meta.pkl"


def _load_faiss():
    global _faiss_index, _faiss_metadata, _faiss_documents
    if not FAISS_AVAILABLE:
        raise RuntimeError("faiss-cpu not installed.")
    idx_path = VECTOR_STORE_CONFIG["faiss_index_path"]
    if os.path.exists(idx_path) and os.path.exists(_FAISS_META_PATH):
        _faiss_index = faiss.read_index(idx_path)
        with open(_FAISS_META_PATH, "rb") as f:
            saved = pickle.load(f)
            _faiss_metadata = saved["metadata"]
            _faiss_documents = saved["documents"]
    else:
        dim = 384  # all-MiniLM-L6-v2 dimension
        _faiss_index = faiss.IndexFlatIP(dim)
        _faiss_metadata = []
        _faiss_documents = []


def _save_faiss():
    idx_path = VECTOR_STORE_CONFIG["faiss_index_path"]
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    faiss.write_index(_faiss_index, idx_path)
    with open(_FAISS_META_PATH, "wb") as f:
        pickle.dump({"metadata": _faiss_metadata, "documents": _faiss_documents}, f)


def ingest_faiss(paper: dict) -> int:
    global _faiss_index, _faiss_metadata, _faiss_documents
    if _faiss_index is None:
        _load_faiss()
    text = (paper.get("abstract") or "") + " " + (paper.get("full_text") or "")
    if not text.strip():
        return 0
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)
    paper_id = _paper_id(paper)
    meta = {
        "paper_id": paper_id,
        "title": paper.get("title", ""),
        "authors": ", ".join(paper.get("authors", [])),
        "year": str(paper.get("year", "")),
        "arxiv_id": paper.get("arxiv_id", ""),
        "doi": paper.get("doi", ""),
    }
    vecs = np.array(embeddings, dtype="float32")
    _faiss_index.add(vecs)
    for chunk in chunks:
        _faiss_metadata.append(meta.copy())
        _faiss_documents.append(chunk)
    _save_faiss()
    return len(chunks)


def retrieve_faiss(query: str, top_k: Optional[int] = None) -> list[dict]:
    global _faiss_index
    if _faiss_index is None:
        _load_faiss()
    k = top_k or EMBEDDING_CONFIG["top_k_retrieval"]
    q_emb = np.array(embed_texts([query]), dtype="float32")
    distances, indices = _faiss_index.search(q_emb, k)
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(_faiss_documents):
            results.append({
                "text": _faiss_documents[idx],
                "metadata": _faiss_metadata[idx],
            })
    return results


# ── Public API (backend-agnostic) ─────────────────────────────────────────────
def ingest_paper(paper: dict) -> int:
    """
    Ingest a paper dict into the configured vector store.
    paper dict keys: title, authors, abstract, full_text, year, arxiv_id, doi
    """
    backend = VECTOR_STORE_CONFIG.get("backend", "chroma")
    if backend == "chroma":
        return ingest_chroma(paper)
    elif backend == "faiss":
        return ingest_faiss(paper)
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")


def retrieve_context(query: str, top_k: Optional[int] = None) -> list[dict]:
    """
    Retrieve top-k relevant chunks for a query.
    Returns list of {"text": str, "metadata": dict}.
    """
    backend = VECTOR_STORE_CONFIG.get("backend", "chroma")
    if backend == "chroma":
        return retrieve_chroma(query, top_k)
    elif backend == "faiss":
        return retrieve_faiss(query, top_k)
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")


def format_context_for_prompt(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a readable context block for Granite.
    """
    if not chunks:
        return "No relevant sources retrieved."
    lines = ["=== Retrieved Source Context ==="]
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        citation = f"{meta.get('authors', 'Unknown')}, {meta.get('year', 'n.d.')}. \"{meta.get('title', 'Untitled')}\""
        lines.append(f"\n[Source {i}] {citation}")
        lines.append(chunk["text"])
    return "\n".join(lines)
