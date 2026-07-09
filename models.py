"""
models.py
---------
SQLAlchemy ORM models for SQLite persistence.
Stores: saved papers, chat history, reference library tags.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class SavedPaper(db.Model):
    __tablename__ = "saved_papers"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.Text, nullable=False)
    authors = db.Column(db.Text, default="")          # JSON-encoded list
    abstract = db.Column(db.Text, default="")
    year = db.Column(db.Integer, nullable=True)
    arxiv_id = db.Column(db.String(64), nullable=True)
    doi = db.Column(db.String(256), nullable=True)
    url = db.Column(db.Text, nullable=True)
    pdf_url = db.Column(db.Text, nullable=True)
    citation_count = db.Column(db.Integer, nullable=True)
    source = db.Column(db.String(32), default="unknown")
    tags = db.Column(db.Text, default="")              # comma-separated tags
    project = db.Column(db.String(128), default="")
    notes = db.Column(db.Text, default="")
    ingested_to_rag = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        import json
        try:
            authors = json.loads(self.authors) if self.authors else []
        except Exception:
            authors = [a.strip() for a in self.authors.split(",") if a.strip()]
        return {
            "id": self.id,
            "title": self.title,
            "authors": authors,
            "abstract": self.abstract,
            "year": self.year,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "citation_count": self.citation_count,
            "source": self.source,
            "tags": [t.strip() for t in self.tags.split(",") if t.strip()],
            "project": self.project,
            "notes": self.notes,
            "ingested_to_rag": self.ingested_to_rag,
            "created_at": self.created_at.isoformat(),
        }


class ChatMessage(db.Model):
    __tablename__ = "chat_messages"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(128), nullable=False, index=True)
    role = db.Column(db.String(16), nullable=False)   # "user" | "assistant"
    content = db.Column(db.Text, nullable=False)
    sources = db.Column(db.Text, default="")           # JSON-encoded source snippets
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        import json
        try:
            sources = json.loads(self.sources) if self.sources else []
        except Exception:
            sources = []
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "sources": sources,
            "created_at": self.created_at.isoformat(),
        }
