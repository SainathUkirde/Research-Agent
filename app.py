"""
app.py
------
Flask application entry point.
Exposes all REST API endpoints for the AI-Powered Research Agent.
All NLG tasks use IBM watsonx.ai + Granite via watsonx_client.py.
"""

import os
import json
import uuid
import logging
from datetime import datetime

from dotenv import load_dotenv
from flask import (
    Flask, request, jsonify, render_template,
    session, abort, send_from_directory
)
from flask_cors import CORS

from models import db, SavedPaper, ChatMessage
from agent_config import AGENT_INSTRUCTIONS, build_system_prompt, GRANITE_CONFIG
from watsonx_client import generate_text, generate_chat, health_check
from rag_pipeline import ingest_paper, retrieve_context, format_context_for_prompt
from paper_apis import (
    search_arxiv, search_semantic_scholar,
    format_citation, format_bibtex
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32).hex())
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///research_agent.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

CORS(app)
db.init_app(app)

with app.app_context():
    db.create_all()
    logger.info("Database tables created / verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────
def _session_id() -> str:
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


def _paper_to_api_dict(paper_dict: dict) -> dict:
    """Normalise a raw paper dict (from APIs) for consistent API responses."""
    return {
        "title": paper_dict.get("title", ""),
        "authors": paper_dict.get("authors", []),
        "abstract": paper_dict.get("abstract", ""),
        "year": paper_dict.get("year"),
        "arxiv_id": paper_dict.get("arxiv_id"),
        "doi": paper_dict.get("doi"),
        "url": paper_dict.get("url"),
        "pdf_url": paper_dict.get("pdf_url"),
        "citation_count": paper_dict.get("citation_count"),
        "source": paper_dict.get("source", "unknown"),
    }


def _err(message: str, status: int = 400):
    return jsonify({"error": message}), status


# ─────────────────────────────────────────────────────────────────────────────
# Page routes (SPA shell)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat")
def chat_page():
    return render_template("chat.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/library")
def library_page():
    return render_template("library.html")


@app.route("/draft")
def draft_page():
    return render_template("draft.html")


@app.route("/hypothesis")
def hypothesis_page():
    return render_template("hypothesis.html")


# ─────────────────────────────────────────────────────────────────────────────
# /api/health
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def api_health():
    """Quick health-check endpoint."""
    result = health_check()
    status_code = 200 if result.get("status") == "ok" else 503
    return jsonify(result), status_code


# ─────────────────────────────────────────────────────────────────────────────
# /api/search — literature search across arXiv + Semantic Scholar
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/search", methods=["POST"])
def api_search():
    """
    POST { "query": str, "sources": ["arxiv", "semantic_scholar"], "max_results": int }
    Returns combined, deduplicated paper list.
    """
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return _err("query is required")

    sources = body.get("sources") or ["arxiv", "semantic_scholar"]
    max_results = int(body.get("max_results") or 8)

    results = []
    seen_titles = set()

    if "arxiv" in sources:
        for p in search_arxiv(query, max_results):
            t = p["title"].lower().strip()
            if t not in seen_titles:
                seen_titles.add(t)
                results.append(_paper_to_api_dict(p))

    if "semantic_scholar" in sources:
        for p in search_semantic_scholar(query, max_results):
            t = p["title"].lower().strip()
            if t not in seen_titles:
                seen_titles.add(t)
                results.append(_paper_to_api_dict(p))

    return jsonify({"papers": results, "count": len(results), "query": query})


# ─────────────────────────────────────────────────────────────────────────────
# /api/chat — RAG-grounded conversational Q&A
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    POST { "message": str, "session_id": str (optional) }
    Returns Granite answer with inline source citations.
    """
    body = request.get_json(silent=True) or {}
    user_msg = (body.get("message") or "").strip()
    if not user_msg:
        return _err("message is required")

    sess_id = body.get("session_id") or _session_id()

    # Retrieve relevant context from vector store
    chunks = retrieve_context(user_msg)
    context_block = format_context_for_prompt(chunks)

    # Load recent chat history (last 6 turns)
    history = (
        ChatMessage.query
        .filter_by(session_id=sess_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(6)
        .all()
    )
    history_msgs = [{"role": m.role, "content": m.content} for m in reversed(history)]

    # Build Granite prompt
    system_prompt = build_system_prompt("conversational research Q&A with RAG context")
    augmented_user_msg = (
        f"Context from retrieved papers:\n{context_block}\n\n"
        f"User question: {user_msg}\n\n"
        "Answer using the provided context. Cite sources inline as [Author, Year]. "
        "If the context is insufficient, say so clearly."
    )
    history_msgs.append({"role": "user", "content": augmented_user_msg})

    try:
        answer = generate_chat(history_msgs, system_prompt=system_prompt)
    except Exception as exc:
        logger.error("Chat generation failed: %s", exc)
        return _err(f"Generation failed: {exc}", 503)

    # Persist messages
    sources_json = json.dumps([
        {
            "text": c["text"][:200],
            "title": c["metadata"].get("title", ""),
            "authors": c["metadata"].get("authors", ""),
            "year": c["metadata"].get("year", ""),
        }
        for c in chunks
    ])
    db.session.add(ChatMessage(session_id=sess_id, role="user", content=user_msg))
    db.session.add(ChatMessage(session_id=sess_id, role="assistant",
                               content=answer, sources=sources_json))
    db.session.commit()

    return jsonify({
        "answer": answer,
        "sources": [
            {
                "text": c["text"][:300],
                "title": c["metadata"].get("title", ""),
                "authors": c["metadata"].get("authors", ""),
                "year": c["metadata"].get("year", ""),
            }
            for c in chunks
        ],
        "session_id": sess_id,
    })


@app.route("/api/chat/history", methods=["GET"])
def api_chat_history():
    """GET /api/chat/history?session_id=xxx — retrieve chat history."""
    sess_id = request.args.get("session_id") or _session_id()
    messages = (
        ChatMessage.query
        .filter_by(session_id=sess_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return jsonify({"messages": [m.to_dict() for m in messages], "session_id": sess_id})


@app.route("/api/chat/clear", methods=["POST"])
def api_chat_clear():
    """POST { "session_id": str } — clear chat history."""
    body = request.get_json(silent=True) or {}
    sess_id = body.get("session_id") or _session_id()
    ChatMessage.query.filter_by(session_id=sess_id).delete()
    db.session.commit()
    return jsonify({"cleared": True, "session_id": sess_id})


# ─────────────────────────────────────────────────────────────────────────────
# /api/summarize — summarize a paper abstract or text
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    """
    POST {
        "text": str,           # abstract or full text
        "title": str,          # optional paper title
        "authors": list[str],  # optional
        "year": int            # optional
    }
    Returns a concise structured summary.
    """
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return _err("text is required")

    title = body.get("title", "")
    authors = ", ".join(body.get("authors") or [])
    year = body.get("year", "")
    max_words = AGENT_INSTRUCTIONS["output_preferences"]["max_summary_length_words"]

    system_prompt = build_system_prompt("paper summarization")
    prompt = (
        f"Summarize the following academic paper in approximately {max_words} words.\n"
        f"Structure the summary as:\n"
        f"  **Problem**: What problem does this paper address?\n"
        f"  **Approach**: What methodology or approach is used?\n"
        f"  **Key Findings**: What are the main results?\n"
        f"  **Significance**: Why does this matter for the field?\n\n"
    )
    if title:
        prompt += f"Title: {title}\n"
    if authors:
        prompt += f"Authors: {authors}\n"
    if year:
        prompt += f"Year: {year}\n"
    prompt += f"\nText:\n{text[:4000]}"

    try:
        summary = generate_text(prompt, system_prompt=system_prompt)
    except Exception as exc:
        return _err(f"Summarization failed: {exc}", 503)

    return jsonify({
        "summary": summary,
        "title": title,
        "authors": body.get("authors", []),
        "year": year,
    })


# ─────────────────────────────────────────────────────────────────────────────
# /api/hypothesis — suggest research hypotheses/gaps from papers
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/hypothesis", methods=["POST"])
def api_hypothesis():
    """
    POST {
        "topic": str,
        "papers": [ { "title": str, "abstract": str, "authors": list, "year": int } ]
    }
    Returns a list of research hypotheses and identified gaps.
    """
    body = request.get_json(silent=True) or {}
    topic = (body.get("topic") or "").strip()
    papers = body.get("papers") or []

    if not topic and not papers:
        return _err("topic or papers list is required")

    # Also retrieve RAG context for the topic
    rag_chunks = retrieve_context(topic) if topic else []
    rag_context = format_context_for_prompt(rag_chunks)

    papers_block = ""
    for i, p in enumerate(papers[:10], 1):
        papers_block += (
            f"\n[Paper {i}] {p.get('title', 'Untitled')} "
            f"({', '.join(p.get('authors', [])[:3])}, {p.get('year', 'n.d.')})\n"
            f"Abstract: {(p.get('abstract') or '')[:500]}\n"
        )

    system_prompt = build_system_prompt("research hypothesis and gap identification")
    prompt = (
        f"Based on the following papers and retrieved context on the topic "
        f"'{topic or 'the provided papers'}', identify:\n\n"
        f"1. **Research Gaps**: Under-explored areas not addressed by current literature\n"
        f"2. **Testable Hypotheses**: Specific, falsifiable hypotheses worth investigating\n"
        f"3. **Suggested Next Steps**: Concrete directions for future research\n\n"
        f"Use bullet points. Reference specific papers where relevant.\n\n"
    )
    if papers_block:
        prompt += f"Provided Papers:\n{papers_block}\n\n"
    if rag_chunks:
        prompt += f"Additional Retrieved Context:\n{rag_context}\n\n"

    try:
        hypothesis_text = generate_text(prompt, system_prompt=system_prompt)
    except Exception as exc:
        return _err(f"Hypothesis generation failed: {exc}", 503)

    return jsonify({
        "hypotheses": hypothesis_text,
        "topic": topic,
        "papers_analyzed": len(papers),
        "rag_sources_used": len(rag_chunks),
    })


# ─────────────────────────────────────────────────────────────────────────────
# /api/draft — draft a paper section from selected sources
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/draft", methods=["POST"])
def api_draft():
    """
    POST {
        "section": "introduction" | "related_work" | "methodology" | "conclusion",
        "topic": str,
        "papers": [ { "title": str, "abstract": str, "authors": list, "year": int } ],
        "instructions": str (optional extra instructions)
    }
    Returns a drafted section with inline citations.
    """
    body = request.get_json(silent=True) or {}
    section = (body.get("section") or "related_work").strip().lower()
    topic = (body.get("topic") or "").strip()
    papers = body.get("papers") or []
    extra_instructions = (body.get("instructions") or "").strip()

    if not topic:
        return _err("topic is required")

    section_map = {
        "introduction": "Introduction section (background, motivation, problem statement, contributions)",
        "related_work": "Related Work section (survey of prior art, comparison, positioning)",
        "methodology": "Methodology section (proposed approach, design decisions, algorithms)",
        "conclusion": "Conclusion section (summary of contributions, limitations, future work)",
    }
    section_desc = section_map.get(section, f"{section} section")

    # Build sources block
    sources_block = ""
    citation_style = AGENT_INSTRUCTIONS["citation_style"]
    for i, p in enumerate(papers[:10], 1):
        from paper_apis import format_citation as fc
        cite = fc(p, style=citation_style)
        sources_block += (
            f"\n[{i}] {cite}\n"
            f"Abstract: {(p.get('abstract') or '')[:400]}\n"
        )

    # Retrieve additional RAG context
    rag_chunks = retrieve_context(topic)
    rag_context = format_context_for_prompt(rag_chunks)

    system_prompt = build_system_prompt(f"drafting academic {section_desc}")
    prompt = (
        f"Draft a {section_desc} for a research paper on the topic: '{topic}'.\n\n"
        f"Requirements:\n"
        f"- Write in academic prose, third person\n"
        f"- Include inline citations in {citation_style} style: [Author, Year]\n"
        f"- Base the draft on the provided sources; do not invent facts\n"
        f"- Aim for 3–5 paragraphs\n"
        f"- Flag any claim not directly supported by the provided sources\n"
    )
    if extra_instructions:
        prompt += f"\nAdditional instructions: {extra_instructions}\n"
    if sources_block:
        prompt += f"\nSources to cite:\n{sources_block}\n"
    if rag_chunks:
        prompt += f"\nAdditional retrieved context:\n{rag_context}\n"

    try:
        draft_text = generate_text(prompt, system_prompt=system_prompt)
    except Exception as exc:
        return _err(f"Draft generation failed: {exc}", 503)

    return jsonify({
        "draft": draft_text,
        "section": section,
        "topic": topic,
        "sources_used": len(papers),
    })


# ─────────────────────────────────────────────────────────────────────────────
# /api/citations — format citations and export BibTeX
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/citations", methods=["POST"])
def api_citations():
    """
    POST {
        "papers": [ paper_dict, ... ],
        "style": "APA" | "MLA" | "IEEE",
        "include_bibtex": bool
    }
    """
    body = request.get_json(silent=True) or {}
    papers = body.get("papers") or []
    style = (body.get("style") or AGENT_INSTRUCTIONS["citation_style"]).upper()
    include_bibtex = bool(body.get("include_bibtex", True))

    if not papers:
        return _err("papers list is required")

    citations = []
    bibtex_entries = []
    for p in papers:
        formatted = format_citation(p, style=style)
        citations.append(formatted)
        if include_bibtex:
            bibtex_entries.append(format_bibtex(p))

    bibtex_block = "\n\n".join(bibtex_entries) if bibtex_entries else ""

    return jsonify({
        "style": style,
        "citations": citations,
        "bibtex": bibtex_block,
        "count": len(citations),
    })


# ─────────────────────────────────────────────────────────────────────────────
# /api/library — saved paper management
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/library", methods=["GET"])
def api_library_list():
    """GET /api/library?project=xxx&tag=yyy — list saved papers."""
    project = request.args.get("project", "").strip()
    tag = request.args.get("tag", "").strip()

    query = SavedPaper.query
    if project:
        query = query.filter(SavedPaper.project == project)
    if tag:
        query = query.filter(SavedPaper.tags.contains(tag))

    papers = query.order_by(SavedPaper.created_at.desc()).all()
    return jsonify({
        "papers": [p.to_dict() for p in papers],
        "count": len(papers),
    })


@app.route("/api/library", methods=["POST"])
def api_library_save():
    """
    POST { paper_dict + optional "tags": str, "project": str, "notes": str }
    Save a paper to the reference library and optionally ingest into RAG.
    """
    body = request.get_json(silent=True) or {}
    title = (body.get("title") or "").strip()
    if not title:
        return _err("title is required")

    # Check for duplicate
    existing = SavedPaper.query.filter_by(title=title).first()
    if existing:
        return jsonify({"saved": False, "message": "Paper already in library", "paper": existing.to_dict()})

    paper = SavedPaper(
        title=title,
        authors=json.dumps(body.get("authors") or []),
        abstract=body.get("abstract") or "",
        year=body.get("year"),
        arxiv_id=body.get("arxiv_id"),
        doi=body.get("doi"),
        url=body.get("url"),
        pdf_url=body.get("pdf_url"),
        citation_count=body.get("citation_count"),
        source=body.get("source", "unknown"),
        tags=body.get("tags", ""),
        project=body.get("project", ""),
        notes=body.get("notes", ""),
    )
    db.session.add(paper)
    db.session.commit()

    # Ingest into RAG vector store
    try:
        ingest_paper(body)
        paper.ingested_to_rag = True
        db.session.commit()
    except Exception as exc:
        logger.warning("RAG ingestion failed for paper '%s': %s", title, exc)

    return jsonify({"saved": True, "paper": paper.to_dict()}), 201


@app.route("/api/library/<int:paper_id>", methods=["GET"])
def api_library_get(paper_id: int):
    paper = db.get_or_404(SavedPaper, paper_id)
    return jsonify(paper.to_dict())


@app.route("/api/library/<int:paper_id>", methods=["PATCH"])
def api_library_update(paper_id: int):
    """PATCH { "tags": str, "project": str, "notes": str } — update metadata."""
    paper = db.get_or_404(SavedPaper, paper_id)
    body = request.get_json(silent=True) or {}
    if "tags" in body:
        paper.tags = body["tags"]
    if "project" in body:
        paper.project = body["project"]
    if "notes" in body:
        paper.notes = body["notes"]
    db.session.commit()
    return jsonify(paper.to_dict())


@app.route("/api/library/<int:paper_id>", methods=["DELETE"])
def api_library_delete(paper_id: int):
    paper = db.get_or_404(SavedPaper, paper_id)
    db.session.delete(paper)
    db.session.commit()
    return jsonify({"deleted": True, "id": paper_id})


@app.route("/api/library/export/bibtex", methods=["GET"])
def api_library_export_bibtex():
    """GET /api/library/export/bibtex?project=xxx — export library as BibTeX file."""
    project = request.args.get("project", "").strip()
    query = SavedPaper.query
    if project:
        query = query.filter_by(project=project)
    papers = query.all()

    entries = []
    for p in papers:
        try:
            authors = json.loads(p.authors) if p.authors else []
        except Exception:
            authors = []
        paper_dict = {
            "title": p.title, "authors": authors, "year": p.year,
            "doi": p.doi, "url": p.url, "abstract": p.abstract,
        }
        entries.append(format_bibtex(paper_dict))

    bibtex_content = "\n\n".join(entries)
    from flask import Response
    return Response(
        bibtex_content,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment; filename=references.bib"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# /api/ingest — manually ingest a paper into the RAG vector store
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """
    POST { paper_dict } — ingest a paper's text into the RAG vector store.
    """
    body = request.get_json(silent=True) or {}
    if not body.get("title") and not body.get("abstract"):
        return _err("At least title or abstract is required")
    try:
        chunks_added = ingest_paper(body)
        return jsonify({"ingested": True, "chunks_added": chunks_added})
    except Exception as exc:
        return _err(f"Ingestion failed: {exc}", 503)


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "endpoint not found"}), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "internal server error"}), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting Research Agent on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
