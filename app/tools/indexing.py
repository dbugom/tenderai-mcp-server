"""Past Proposal Indexing tools — parse, save, and search past proposals.

Data-tool pattern: index_past_proposal parses files and returns raw text.
Claude structures the metadata, then calls save_proposal_index to store it.

Supports two search modes:
- FTS5: keyword search with BM25 ranking (always available)
- Hybrid: FTS5 + vector similarity via sqlite-vec + Voyage AI embeddings,
  combined with Reciprocal Rank Fusion (enabled when VOYAGE_API_KEY is set)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from app.db.database import Database
from app.services.parser import ParserService

logger = logging.getLogger(__name__)

# File extensions we can parse
INDEXABLE_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".md", ".txt"}

# Budget for combined text returned to Claude
MAX_TOTAL_CHARS = 25_000
FINANCIAL_RESERVE_CHARS = 8_000

# Reciprocal Rank Fusion constant (standard value from literature)
RRF_K = 60


def _rrf_combine(
    fts_results: list[dict],
    vec_results: list[dict],
    fts_weight: float = 1.0,
    vec_weight: float = 1.0,
) -> list[dict]:
    """Combine FTS5 and vector search results using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, doc in enumerate(fts_results):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0) + fts_weight / (RRF_K + rank + 1)
        docs[doc_id] = doc

    for rank, doc in enumerate(vec_results):
        doc_id = doc["id"]
        scores[doc_id] = scores.get(doc_id, 0) + vec_weight / (RRF_K + rank + 1)
        if doc_id not in docs:
            docs[doc_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**docs[doc_id], "rrf_score": score}
        for doc_id, score in ranked
    ]


def register_indexing_tools(
    mcp: FastMCP,
    db: Database,
    parser: ParserService,
    data_dir: Path,
    embeddings=None,
) -> None:
    """Register past proposal indexing and search tools on the MCP server."""

    past_dir = data_dir / "past_proposals"

    async def _read_file(file_path: Path) -> str:
        """Read a single file, using parser for binary formats."""
        ext = file_path.suffix.lower()
        if ext in (".md", ".txt"):
            return file_path.read_text()
        elif ext in (".pdf", ".docx", ".doc", ".xlsx", ".xls"):
            try:
                parsed = await parser.parse_file(str(file_path))
                return parsed["text"]
            except Exception as e:
                logger.warning("Could not parse %s: %s", file_path, e)
                return ""
        return ""

    @mcp.tool()
    async def index_past_proposal(folder_name: str) -> dict:
        """Parse all files in a past proposal folder and return raw content for analysis.

        Scans the folder, parses all supported files (PDF, DOCX, XLSX, MD, TXT),
        and returns the combined content. You (Claude) should analyze the content,
        extract structured metadata, then call save_proposal_index to store it.

        Args:
            folder_name: Name of the folder inside data/past_proposals/

        Returns:
            Dict with folder_name, file_list, combined_text (with financial data prioritized),
            and file_count
        """
        folder_path = past_dir / folder_name
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(
                f"Folder not found: {folder_path}. "
                f"Expected a directory inside data/past_proposals/"
            )

        # Collect all parseable files, skipping _-prefixed files
        files = sorted(
            f for f in folder_path.iterdir()
            if f.is_file()
            and f.suffix.lower() in INDEXABLE_EXTENSIONS
            and not f.name.startswith("_")
        )
        if not files:
            raise ValueError(f"No parseable files found in {folder_path}")

        file_list = [f.name for f in files]
        logger.info("Parsing %d files from %s", len(files), folder_name)

        # Parse files, separating financial (XLSX) from others
        financial_texts = []
        other_texts = []
        for f in files:
            content = await _read_file(f)
            if not content:
                continue
            if f.suffix.lower() in (".xlsx", ".xls"):
                financial_texts.append(f"=== {f.name} (Financial) ===\n{content}")
            else:
                other_texts.append(f"=== {f.name} ===\n{content}")

        # Build combined text with financial data prioritized
        financial_combined = "\n\n".join(financial_texts)
        other_combined = "\n\n".join(other_texts)

        if financial_combined:
            financial_combined = financial_combined[:FINANCIAL_RESERVE_CHARS]
            remaining = MAX_TOTAL_CHARS - len(financial_combined)
            other_combined = other_combined[:max(remaining, 5000)]
        else:
            other_combined = other_combined[:MAX_TOTAL_CHARS]

        combined_text = other_combined
        if financial_combined:
            combined_text += "\n\n--- FINANCIAL DATA ---\n\n" + financial_combined

        return {
            "folder_name": folder_name,
            "file_list": file_list,
            "file_count": len(files),
            "combined_text": combined_text,
            "instructions": (
                "Analyze the above content and extract structured metadata as JSON with these fields:\n"
                "tender_number, title, client, sector (telecom|it|infrastructure|security|energy|general), "
                "country (2-letter code), technical_summary, pricing_summary, total_price (float), "
                "margin_info, technologies (list), keywords (10-20 items), full_summary.\n"
                "Then call save_proposal_index with the extracted data."
            ),
        }

    @mcp.tool()
    async def save_proposal_index(
        folder_name: str,
        title: str,
        client: str = "",
        sector: str = "",
        country: str = "",
        tender_number: str = "",
        technical_summary: str = "",
        pricing_summary: str = "",
        total_price: float = 0.0,
        margin_info: str = "",
        technologies: Optional[list] = None,
        keywords: Optional[list] = None,
        full_summary: str = "",
    ) -> dict:
        """Save structured proposal metadata extracted by Claude into the search index.

        Call this after analyzing the raw content from index_past_proposal.
        Generates _summary.md file and indexes for FTS5/vector search.

        Args:
            folder_name: Name of the folder inside data/past_proposals/
            title: Full tender/project title
            client: Issuing organization name
            sector: One of: telecom, it, infrastructure, security, energy, general
            country: Two-letter country code
            tender_number: RFP/tender reference number
            technical_summary: 2-3 paragraph summary of the technical solution
            pricing_summary: Summary of pricing structure
            total_price: Total project price
            margin_info: Margin percentages if found
            technologies: List of specific products/vendors/technologies
            keywords: 10-20 searchable keywords
            full_summary: Comprehensive summary of the entire proposal

        Returns:
            Dict with index_id, folder_name, title, and vector_indexed status
        """
        folder_path = past_dir / folder_name
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")

        techs = technologies or []
        kws = keywords or []

        # Count files
        files = [
            f for f in folder_path.iterdir()
            if f.is_file()
            and f.suffix.lower() in INDEXABLE_EXTENSIONS
            and not f.name.startswith("_")
        ]
        file_list = [f.name for f in files]

        # Build _summary.md for human readability
        summary_md = (
            f"# {title}\n\n"
            f"**Client:** {client}\n"
            f"**Sector:** {sector}\n"
            f"**Country:** {country}\n"
            f"**Tender Number:** {tender_number}\n\n"
            f"## Technical Summary\n{technical_summary}\n\n"
            f"## Pricing Summary\n{pricing_summary}\n"
            f"**Total Price:** {total_price}\n"
            f"**Margin Info:** {margin_info}\n\n"
            f"## Technologies\n"
            + "\n".join(f"- {t}" for t in techs)
            + f"\n\n## Keywords\n"
            + ", ".join(kws)
            + f"\n\n## Full Summary\n{full_summary}\n"
        )
        summary_path = folder_path / "_summary.md"
        summary_path.write_text(summary_md)
        logger.info("Wrote summary to %s", summary_path)

        # Upsert into database (triggers auto-sync FTS5)
        index_record = await db.upsert_proposal_index(
            folder_name=folder_name,
            tender_number=tender_number,
            title=title,
            client=client,
            sector=sector,
            country=country,
            technical_summary=technical_summary,
            pricing_summary=pricing_summary,
            total_price=total_price,
            margin_info=margin_info,
            technologies=techs,
            keywords=kws,
            full_summary=full_summary,
            file_count=len(files),
            file_list=file_list,
        )

        logger.info("Indexed proposal '%s' (id=%s)", folder_name, index_record["id"])

        # Generate and store vector embedding if available
        vector_stored = False
        if embeddings and db.vec_enabled:
            try:
                embed_text = (
                    f"{title} {client} {sector} "
                    f"{technical_summary} "
                    f"{' '.join(kws)} "
                    f"{full_summary}"
                )
                vector = await embeddings.embed(embed_text[:8000])
                vector_stored = await db.upsert_proposal_vector(folder_name, vector)
                if vector_stored:
                    logger.info("Stored embedding vector for '%s'", folder_name)
            except Exception as e:
                logger.warning("Could not generate/store embedding for %s: %s", folder_name, e)

        return {
            "index_id": index_record["id"],
            "folder_name": folder_name,
            "title": title,
            "client": client,
            "sector": sector,
            "file_count": len(files),
            "technologies": techs,
            "summary_path": str(summary_path),
            "vector_indexed": vector_stored,
        }

    @mcp.tool()
    async def search_past_proposals(
        query: str, sector: str = "", limit: int = 5, mode: str = "auto"
    ) -> dict:
        """Search indexed past proposals using keyword, semantic, or hybrid search.

        Modes:
        - "auto": Uses hybrid (FTS5 + vector RRF) if embeddings are available, otherwise FTS5-only
        - "keyword": FTS5 only — supports quoted phrases, prefix (cisco*), boolean (AND/OR)
        - "semantic": Vector similarity only — finds conceptually similar proposals
        - "hybrid": Combines FTS5 + vector using Reciprocal Rank Fusion

        Args:
            query: Search query text
            sector: Optional sector filter
            limit: Maximum results to return (default 5)
            mode: Search mode — "auto", "keyword", "semantic", or "hybrid"

        Returns:
            Dict with matches (ranked list), result_count, and search_mode used
        """
        has_vec = embeddings is not None and db.vec_enabled

        if mode == "auto":
            actual_mode = "hybrid" if has_vec else "keyword"
        elif mode in ("semantic", "hybrid") and not has_vec:
            actual_mode = "keyword"
            logger.info("Vector search unavailable, falling back to keyword mode")
        else:
            actual_mode = mode

        fts_results = []
        vec_results = []

        if actual_mode in ("keyword", "hybrid"):
            try:
                fts_results = await db.search_proposal_index(
                    query, sector=sector, limit=limit * 2
                )
            except Exception as e:
                logger.warning("FTS5 search failed: %s", e)

        if actual_mode in ("semantic", "hybrid") and has_vec:
            try:
                query_vec = await embeddings.embed_query(query)
                vec_results = await db.search_proposal_vector(
                    query_vec, limit=limit * 2
                )
                if sector and vec_results:
                    vec_results = [
                        r for r in vec_results
                        if r.get("sector", "").lower() == sector.lower()
                    ]
            except Exception as e:
                logger.warning("Vector search failed: %s", e)

        if actual_mode == "hybrid" and fts_results and vec_results:
            combined = _rrf_combine(fts_results, vec_results)[:limit]
        elif actual_mode == "semantic" and vec_results:
            combined = vec_results[:limit]
        else:
            combined = fts_results[:limit]

        matches = []
        for r in combined:
            matches.append({
                "index_id": r["id"],
                "folder_name": r["folder_name"],
                "title": r.get("title", ""),
                "client": r.get("client", ""),
                "sector": r.get("sector", ""),
                "country": r.get("country", ""),
                "technical_summary": r.get("technical_summary", ""),
                "pricing_summary": r.get("pricing_summary", ""),
                "total_price": r.get("total_price", 0.0),
                "technologies": r.get("technologies", []),
                "rrf_score": r.get("rrf_score"),
                "distance": r.get("distance"),
                "rank": r.get("rank"),
            })

        return {
            "query": query,
            "sector_filter": sector,
            "search_mode": actual_mode,
            "vector_available": has_vec,
            "result_count": len(matches),
            "matches": matches,
        }

    @mcp.tool()
    async def list_indexed_proposals() -> dict:
        """List all indexed past proposals with aggregate statistics.

        Returns:
            Dict with proposals list, total count, and breakdowns by sector, country, and total value
        """
        rows = await db.list_proposal_indexes()

        proposals = []
        by_sector: dict[str, int] = {}
        by_country: dict[str, int] = {}
        total_value = 0.0

        for r in rows:
            proposals.append({
                "index_id": r["id"],
                "folder_name": r["folder_name"],
                "title": r.get("title", ""),
                "client": r.get("client", ""),
                "sector": r.get("sector", ""),
                "country": r.get("country", ""),
                "total_price": r.get("total_price", 0.0),
                "file_count": r.get("file_count", 0),
                "technologies": r.get("technologies", []),
                "indexed_at": r.get("indexed_at", ""),
            })
            sector = r.get("sector", "unknown")
            by_sector[sector] = by_sector.get(sector, 0) + 1
            country = r.get("country", "unknown")
            by_country[country] = by_country.get(country, 0) + 1
            total_value += r.get("total_price", 0.0)

        return {
            "total_count": len(proposals),
            "by_sector": by_sector,
            "by_country": by_country,
            "total_value": total_value,
            "vector_search_available": embeddings is not None and db.vec_enabled,
            "proposals": proposals,
        }
