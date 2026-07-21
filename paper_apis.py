"""
paper_apis.py
-------------
Integration with academic paper APIs:
  - arXiv API (via the `arxiv` Python library)
  - Semantic Scholar API (REST, no key required for basic use)

All functions return a unified paper dict schema:
{
    "title": str,
    "authors": list[str],
    "abstract": str,
    "year": int | None,
    "arxiv_id": str | None,
    "doi": str | None,
    "url": str | None,
    "citation_count": int | None,
    "source": "arxiv" | "semantic_scholar",
    "pdf_url": str | None,
}
"""

import logging
import requests
from typing import Optional

from agent_config import API_CONFIG

logger = logging.getLogger(__name__)

TIMEOUT = API_CONFIG["request_timeout_seconds"]


# ── arXiv ─────────────────────────────────────────────────────────────────────
try:
    import arxiv as arxiv_lib
    ARXIV_LIB_AVAILABLE = True
except ImportError:
    ARXIV_LIB_AVAILABLE = False
    logger.warning("arxiv library not installed.  arXiv search disabled.")


def search_arxiv(query: str, max_results: Optional[int] = None) -> list[dict]:
    """
    Search arXiv for papers matching *query*.
    Returns a list of unified paper dicts.
    """
    if not ARXIV_LIB_AVAILABLE:
        logger.error("arxiv library not available.")
        return []

    limit = max_results or API_CONFIG["arxiv_max_results"]
    try:
        client = arxiv_lib.Client()
        search = arxiv_lib.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv_lib.SortCriterion.Relevance,
        )
        papers = []
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "authors": [str(a) for a in result.authors],
                "abstract": result.summary.replace("\n", " "),
                "year": result.published.year if result.published else None,
                "arxiv_id": result.entry_id.split("/")[-1],
                "doi": result.doi,
                "url": result.entry_id,
                "citation_count": None,
                "source": "arxiv",
                "pdf_url": result.pdf_url,
            })
        logger.info("arXiv returned %d results for query: %s", len(papers), query)
        return papers
    except Exception as exc:
        logger.error("arXiv search failed: %s", exc)
        return []


def get_arxiv_paper(arxiv_id: str) -> Optional[dict]:
    """Fetch a single arXiv paper by its ID."""
    if not ARXIV_LIB_AVAILABLE:
        return None
    try:
        client = arxiv_lib.Client()
        search = arxiv_lib.Search(id_list=[arxiv_id])
        for result in client.results(search):
            return {
                "title": result.title,
                "authors": [str(a) for a in result.authors],
                "abstract": result.summary.replace("\n", " "),
                "year": result.published.year if result.published else None,
                "arxiv_id": arxiv_id,
                "doi": result.doi,
                "url": result.entry_id,
                "citation_count": None,
                "source": "arxiv",
                "pdf_url": result.pdf_url,
            }
    except Exception as exc:
        logger.error("arXiv fetch failed for %s: %s", arxiv_id, exc)
    return None


# ── Semantic Scholar ──────────────────────────────────────────────────────────
SS_BASE = API_CONFIG["semantic_scholar_base_url"]
SS_FIELDS = "title,authors,year,abstract,externalIds,citationCount,url,openAccessPdf"


def search_semantic_scholar(query: str, max_results: Optional[int] = None) -> list[dict]:
    """
    Search Semantic Scholar for papers matching *query*.
    Returns a list of unified paper dicts.
    """
    limit = max_results or API_CONFIG["semantic_scholar_max_results"]
    url = f"{SS_BASE}/paper/search"
    params = {"query": query, "limit": limit, "fields": SS_FIELDS}
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        papers = []
        for item in data.get("data", []):
            external = item.get("externalIds") or {}
            pdf_info = item.get("openAccessPdf") or {}
            authors = [a.get("name", "") for a in item.get("authors", [])]
            papers.append({
                "title": item.get("title", ""),
                "authors": authors,
                "abstract": item.get("abstract") or "",
                "year": item.get("year"),
                "arxiv_id": external.get("ArXiv"),
                "doi": external.get("DOI"),
                "url": item.get("url"),
                "citation_count": item.get("citationCount"),
                "source": "semantic_scholar",
                "pdf_url": pdf_info.get("url"),
            })
        logger.info("Semantic Scholar returned %d results for: %s", len(papers), query)
        return papers
    except Exception as exc:
        logger.error("Semantic Scholar search failed: %s", exc)
        return []


def get_semantic_scholar_paper(paper_id: str) -> Optional[dict]:
    """
    Fetch a single paper from Semantic Scholar by its paper ID or DOI.
    paper_id can be an S2 paper ID, DOI:xxx, ArXiv:xxx, etc.
    """
    url = f"{SS_BASE}/paper/{paper_id}"
    params = {"fields": SS_FIELDS}
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        item = resp.json()
        external = item.get("externalIds") or {}
        pdf_info = item.get("openAccessPdf") or {}
        authors = [a.get("name", "") for a in item.get("authors", [])]
        return {
            "title": item.get("title", ""),
            "authors": authors,
            "abstract": item.get("abstract") or "",
            "year": item.get("year"),
            "arxiv_id": external.get("ArXiv"),
            "doi": external.get("DOI"),
            "url": item.get("url"),
            "citation_count": item.get("citationCount"),
            "source": "semantic_scholar",
            "pdf_url": pdf_info.get("url"),
        }
    except Exception as exc:
        logger.error("Semantic Scholar fetch failed for %s: %s", paper_id, exc)
    return None


# ── Citation formatters ───────────────────────────────────────────────────────
def _author_last_first(name: str) -> str:
    parts = name.strip().split()
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return name


def format_apa(paper: dict) -> str:
    authors = paper.get("authors", [])
    if not authors:
        author_str = "Unknown Author"
    elif len(authors) == 1:
        author_str = _author_last_first(authors[0])
    elif len(authors) <= 7:
        formatted = [_author_last_first(a) for a in authors]
        author_str = ", ".join(formatted[:-1]) + ", & " + formatted[-1]
    else:
        formatted = [_author_last_first(a) for a in authors[:6]]
        author_str = ", ".join(formatted) + ", ... " + _author_last_first(authors[-1])
    year = paper.get("year") or "n.d."
    title = paper.get("title", "Untitled")
    doi = paper.get("doi", "")
    doi_str = f" https://doi.org/{doi}" if doi else (paper.get("url") or "")
    return f"{author_str} ({year}). {title}.{doi_str}"


def format_mla(paper: dict) -> str:
    authors = paper.get("authors", [])
    if not authors:
        author_str = "Unknown Author"
    elif len(authors) == 1:
        author_str = _author_last_first(authors[0])
    elif len(authors) == 2:
        author_str = f"{_author_last_first(authors[0])}, and {authors[1]}"
    else:
        author_str = f"{_author_last_first(authors[0])}, et al."
    year = paper.get("year") or "n.d."
    title = paper.get("title", "Untitled")
    doi = paper.get("doi", "")
    doi_str = f" doi:{doi}." if doi else ""
    return f'{author_str}. "{title}." {year}.{doi_str}'


def format_ieee(paper: dict) -> str:
    authors = paper.get("authors", [])

    def initials(name):
        parts = name.strip().split()
        if len(parts) >= 2:
            first_initials = " ".join(p[0] + "." for p in parts[:-1])
            return f"{first_initials} {parts[-1]}"
        return name

    if not authors:
        author_str = "Unknown Author"
    elif len(authors) <= 3:
        author_str = ", ".join(initials(a) for a in authors)
    else:
        author_str = initials(authors[0]) + " et al."
    year = paper.get("year") or "n.d."
    title = paper.get("title", "Untitled")
    doi = paper.get("doi", "")
    doi_str = f", doi: {doi}" if doi else ""
    return f'{author_str}, "{title}," {year}{doi_str}.'


def format_bibtex(paper: dict) -> str:
    first_author = (paper.get("authors") or ["Unknown"])[0].split()[-1].lower()
    year = paper.get("year") or "0000"
    title_word = (paper.get("title") or "paper").split()[0].lower()
    key = f"{first_author}{year}{title_word}"
    authors_bib = " and ".join(paper.get("authors") or ["Unknown Author"])
    doi_line = f"  doi = {{{paper['doi']}}},\n" if paper.get("doi") else ""
    url_line = f"  url = {{{paper['url']}}},\n" if paper.get("url") else ""
    abstract = (paper.get("abstract") or "").replace("{", "").replace("}", "")[:300]
    return (
        f"@article{{{key},\n"
        f"  title = {{{paper.get('title', 'Untitled')}}},\n"
        f"  author = {{{authors_bib}}},\n"
        f"  year = {{{year}}},\n"
        f"{doi_line}"
        f"{url_line}"
        f"  abstract = {{{abstract}}},\n"
        f"}}"
    )


def format_citation(paper: dict, style: str = "APA") -> str:
    style = style.upper()
    if style == "APA":
        return format_apa(paper)
    elif style == "MLA":
        return format_mla(paper)
    elif style == "IEEE":
        return format_ieee(paper)
    elif style == "BIBTEX":
        return format_bibtex(paper)
    return format_apa(paper)
