"""Past Proposal Indexing tools — parse, summarize, and search past proposals.

Supports two search modes:
- FTS5: keyword search with BM25 ranking (always available)
- Hybrid: FTS5 + vector similarity via sqlite-vec + Voyage AI embeddings,
  combined with Reciprocal Rank Fusion (enabled when VOYAGE_API_KEY is set)

Supports two indexing modes:
- With ANTHROPIC_API_KEY: server-side LLM extracts structured metadata automatically
- Without ANTHROPIC_API_KEY (data-tool pattern): parses files and returns raw text
  for Claude to analyze, then saves via save_proposal_index tool
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from app.db.database import Database
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.parser import ParserService

logger = logging.getLogger(__name__)

# File extensions we can parse
INDEXABLE_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".md", ".txt"}

# Budget for combined text sent to LLM / returned to Claude
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
    """Combine FTS5 and vector search results using Reciprocal Rank Fusion.

    RRF score = sum(weight / (k + rank)) across both result lists.
    Higher score = better match.
    """
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

    # Sort by combined RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**docs[doc_id], "rrf_score": score}
        for doc_id, score in ranked
    ]


def register_indexing_tools(
    mcp: FastMCP,
    db: Database,
    llm: Optional[LLMService],
    parser: ParserService,
    data_dir: Path,
    embeddings: Optional[EmbeddingService] = None,
) -> None:
    """Register past proposal indexing and search tools on the MCP server."""

    past_dir = data_dir / "past_proposals"

    # Check if LLM is usable (has API key)
    has_llm = llm is not None and bool(getattr(llm, 'client', None))
    try:
        if has_llm and not llm.client.api_key:
            has_llm = False
    except Exception:
        pass

    if has_llm:
        logger.info("Indexing tools: LLM available — server-side extraction enabled")
    else:
        logger.info("Indexing tools: No LLM — data-tool mode (Claude does analysis)")

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

    async def _parse_folder_files(folder_name: str, folder_path: Path) -> dict:
        """Parse all files in a folder and return combined text + file list."""
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

        # Truncate: reserve space for financial data
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
        }

    async def _save_extracted(
        folder_name: str,
        folder_path: Path,
        extracted: dict,
        file_count: int,
        file_list: list[str],
    ) -> dict:
        """Save extracted metadata to DB, write _summary.md, generate embedding."""
        # Build _summary.md for human readability
        summary_md = (
            f"# {extracted.get('title', folder_name)}\n\n"
            f"**Client:** {extracted.get('client', 'Unknown')}\n"
            f"**Sector:** {extracted.get('sector', '')}\n"
            f"**Country:** {extracted.get('country', '')}\n"
            f"**Tender Number:** {extracted.get('tender_number', '')}\n\n"
            f"## Technical Summary\n{extracted.get('technical_summary', '')}\n\n"
            f"## Pricing Summary\n{extracted.get('pricing_summary', '')}\n"
            f"**Total Price:** {extracted.get('total_price', 0.0)}\n"
            f"**Margin Info:** {extracted.get('margin_info', '')}\n\n"
            f"## Technologies\n"
            + "\n".join(f"- {t}" for t in extracted.get("technologies", []))
            + f"\n\n## Keywords\n"
            + ", ".join(extracted.get("keywords", []))
            + f"\n\n## Full Summary\n{extracted.get('full_summary', '')}\n"
        )
        summary_path = folder_path / "_summary.md"
        summary_path.write_text(summary_md)
        logger.info("Wrote summary to %s", summary_path)

        # Upsert into database (triggers auto-sync FTS5)
        index_record = await db.upsert_proposal_index(
            folder_name=folder_name,
            tender_number=extracted.get("tender_number", ""),
            title=extracted.get("title", folder_name),
            client=extracted.get("client", ""),
            sector=extracted.get("sector", ""),
            country=extracted.get("country", ""),
            technical_summary=extracted.get("technical_summary", ""),
            pricing_summary=extracted.get("pricing_summary", ""),
            total_price=float(extracted.get("total_price", 0.0)),
            margin_info=extracted.get("margin_info", ""),
            technologies=extracted.get("technologies", []),
            keywords=extracted.get("keywords", []),
            full_summary=extracted.get("full_summary", ""),
            file_count=file_count,
            file_list=file_list,
        )

        logger.info("Indexed proposal '%s' (id=%s)", folder_name, index_record["id"])

        # Generate and store vector embedding if available
        vector_stored = False
        if embeddings and db.vec_enabled:
            try:
                embed_text = (
                    f"{extracted.get('title', '')} "
                    f"{extracted.get('client', '')} "
                    f"{extracted.get('sector', '')} "
                    f"{extracted.get('technical_summary', '')} "
                    f"{' '.join(extracted.get('keywords', []))} "
                    f"{extracted.get('full_summary', '')}"
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
            "title": index_record.get("title", ""),
            "client": index_record.get("client", ""),
            "sector": index_record.get("sector", ""),
            "file_count": file_count,
            "technologies": index_record.get("technologies", []),
            "summary_path": str(summary_path),
            "vector_indexed": vector_stored,
        }

    async def _index_single_folder_with_llm(folder_name: str, folder_path: Path) -> dict:
        """Index a single proposal folder using server-side LLM extraction."""
        parsed = await _parse_folder_files(folder_name, folder_path)

        # Call LLM for structured extraction
        user_prompt = (
            f"Analyze the following past proposal documents from folder '{folder_name}' "
            f"and extract structured metadata.\n\n"
            f"Files: {', '.join(parsed['file_list'])}\n\n"
            f"{parsed['combined_text']}"
        )

        raw_response = await llm.generate_section(
            "proposal_summary", user_prompt, max_tokens=4096
        )

        # Parse JSON response (handle ```json blocks)
        json_text = raw_response.strip()
        if json_text.startswith("```"):
            lines = json_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_text = "\n".join(lines)

        try:
            extracted = json.loads(json_text)
        except json.JSONDecodeError:
            logger.error("LLM returned invalid JSON for %s: %s", folder_name, json_text[:200])
            extracted = {
                "title": folder_name,
                "full_summary": raw_response[:1000],
            }

        return await _save_extracted(
            folder_name, folder_path, extracted,
            parsed["file_count"], parsed["file_list"],
        )

    @mcp.tool()
    async def index_past_proposal(
        folder_name: str,
        batch_size: int = 10,
        skip_already_indexed: bool = True,
    ) -> dict:
        """Parse all files in a past proposal folder and index for fast search.

        Scans the folder, parses all supported files (PDF, DOCX, XLSX, MD, TXT).

        If the server has ANTHROPIC_API_KEY configured, extraction happens automatically.
        Otherwise, returns the parsed text for each folder so you (Claude) can analyze it
        and call save_proposal_index to store the structured metadata.

        Supports nested folders: if the folder contains subdirectories instead of
        parseable files, each subdirectory is processed as a separate proposal.
        For example, folder_name="TRA" will process all subfolders inside TRA/.

        For large parent folders, processes in batches. Call this tool repeatedly
        with the same folder_name until progress shows 100% complete.

        Args:
            folder_name: Name of the folder inside data/past_proposals/
                         Can be a single proposal folder or a parent folder
                         containing multiple proposal subfolders.
            batch_size: Number of subfolders to process per call (default 10).
                        Only applies to parent folders with subdirectories.
            skip_already_indexed: Skip subfolders that have already been indexed
                                  (default True). Set to False to re-index all.

        Returns:
            If server has LLM: indexed results with metadata.
            If no LLM: parsed text data for each folder — call save_proposal_index
            with your analysis for each folder to complete indexing.
            Batch results include progress_percent and remaining count.
        """
        folder_path = past_dir / folder_name
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(
                f"Folder not found: {folder_path}. "
                f"Expected a directory inside data/past_proposals/"
            )

        # Check if folder has parseable files directly (single proposal)
        direct_files = [
            f for f in folder_path.iterdir()
            if f.is_file()
            and f.suffix.lower() in INDEXABLE_EXTENSIONS
            and not f.name.startswith("_")
        ]

        if direct_files:
            # Single proposal folder
            if has_llm:
                return await _index_single_folder_with_llm(folder_name, folder_path)
            else:
                parsed = await _parse_folder_files(folder_name, folder_path)
                return {
                    "mode": "data_tool",
                    "needs_save": True,
                    "message": (
                        "Parsed files successfully. Analyze the text below and call "
                        "save_proposal_index with the extracted metadata."
                    ),
                    **parsed,
                }

        # --- Nested folder: batch processing ---
        all_subdirs = sorted(
            d for d in folder_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        if not all_subdirs:
            raise ValueError(
                f"No parseable files or subdirectories found in {folder_path}"
            )

        total = len(all_subdirs)

        # Determine which folders still need indexing
        already_indexed = []
        pending_subdirs = []

        for subdir in all_subdirs:
            sub_name = f"{folder_name}/{subdir.name}"
            if skip_already_indexed:
                existing = await db.get_proposal_index_by_folder(sub_name)
                if existing:
                    already_indexed.append(sub_name)
                    continue
            pending_subdirs.append(subdir)

        previously_done = len(already_indexed)

        if not pending_subdirs:
            return {
                "batch": True,
                "parent_folder": folder_name,
                "total_subfolders": total,
                "previously_indexed": previously_done,
                "indexed_this_batch": 0,
                "failed": 0,
                "skipped": 0,
                "progress_percent": 100,
                "complete": True,
                "message": f"All {total} subfolders already indexed. Nothing to do.",
            }

        # Process only batch_size folders in this call
        batch = pending_subdirs[:batch_size]

        logger.info(
            "Batch processing %d of %d pending subfolders in %s "
            "(%d already indexed, mode=%s)",
            len(batch), len(pending_subdirs), folder_name,
            previously_done, "llm" if has_llm else "data_tool",
        )

        results = []
        parsed_items = []  # For data-tool mode
        failed = []
        skipped = []

        for i, subdir in enumerate(batch, 1):
            sub_name = f"{folder_name}/{subdir.name}"
            logger.info(
                "Processing [%d/%d in batch, %d/%d overall]: %s",
                i, len(batch),
                previously_done + len(results) + len(parsed_items) + len(skipped) + len(failed) + 1,
                total,
                sub_name,
            )
            try:
                if has_llm:
                    result = await _index_single_folder_with_llm(sub_name, subdir)
                    results.append(result)
                else:
                    parsed = await _parse_folder_files(sub_name, subdir)
                    parsed_items.append(parsed)
            except ValueError as e:
                skipped.append({"folder": sub_name, "reason": str(e)})
                logger.info("Skipped %s: %s", sub_name, e)
            except Exception as e:
                failed.append({"folder": sub_name, "error": str(e)})
                logger.error("Failed to process %s: %s", sub_name, e)

        done_after = previously_done + len(results) + len(skipped)
        # In data-tool mode, parsed_items are not "done" until save_proposal_index is called
        if has_llm:
            remaining = total - done_after - len(failed)
        else:
            remaining = len(pending_subdirs) - len(parsed_items) - len(skipped) - len(failed)
            done_after = previously_done + len(skipped)

        progress_pct = round(
            (previously_done + len(results) + len(skipped)) / total * 100, 1
        )
        complete = has_llm and remaining <= 0

        response: dict = {
            "batch": True,
            "parent_folder": folder_name,
            "total_subfolders": total,
            "previously_indexed": previously_done,
            "failed": len(failed),
            "skipped": len(skipped),
            "progress_percent": progress_pct,
            "complete": complete,
            "failures": failed,
            "skipped_details": skipped,
        }

        if has_llm:
            response.update({
                "mode": "llm",
                "indexed_this_batch": len(results),
                "done_so_far": done_after,
                "remaining": max(remaining, 0),
                "message": (
                    f"Batch complete: indexed {len(results)}, "
                    f"skipped {len(skipped)}, failed {len(failed)}. "
                    f"Overall progress: {done_after}/{total} ({progress_pct}%). "
                    + ("All done!" if complete else f"{remaining} remaining — call again to continue.")
                ),
                "results": results,
            })
        else:
            response.update({
                "mode": "data_tool",
                "needs_save": True,
                "parsed_this_batch": len(parsed_items),
                "remaining": max(remaining, 0),
                "message": (
                    f"Parsed {len(parsed_items)} folders (skipped {len(skipped)}, "
                    f"failed {len(failed)}). "
                    f"Previously indexed: {previously_done}/{total}. "
                    f"Analyze each parsed folder below and call save_proposal_index "
                    f"for each one. Then call index_past_proposal again to continue "
                    f"with the next batch."
                ),
                "parsed_folders": parsed_items,
            })

        return response

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
        technologies: list[str] | None = None,
        keywords: list[str] | None = None,
        full_summary: str = "",
    ) -> dict:
        """Save structured metadata for a past proposal after analysis.

        Call this after index_past_proposal returns parsed text in data-tool mode.
        Analyze the parsed text, extract the metadata fields, and pass them here
        to store in the database, generate the _summary.md file, and create
        vector embeddings for search.

        Args:
            folder_name: Exact folder_name as returned by index_past_proposal
                         (e.g., "TRA/TRA000208-Cisco Switches")
            title: Full tender/project title
            client: Issuing organization name
            sector: One of: telecom, it, infrastructure, security, energy, general
            country: Two-letter country code (e.g., OM, AE, SA)
            tender_number: RFP/tender reference number
            technical_summary: 2-3 paragraph summary of the technical solution
            pricing_summary: Summary of pricing structure and key cost components
            total_price: Total price as a number
            margin_info: Margin percentages if found
            technologies: List of specific products, vendors, technologies mentioned
            keywords: 10-20 searchable keywords covering scope, sector, tech
            full_summary: Comprehensive 3-5 paragraph summary of the entire proposal

        Returns:
            Dict with index_id, folder_name, title, and confirmation details
        """
        techs = technologies or []
        kws = keywords or []

        # Resolve the folder path
        folder_path = past_dir / folder_name
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(
                f"Folder not found: {folder_path}. "
                f"The folder_name must match exactly what index_past_proposal returned."
            )

        # Count actual files for the record
        files = sorted(
            f for f in folder_path.iterdir()
            if f.is_file()
            and f.suffix.lower() in INDEXABLE_EXTENSIONS
            and not f.name.startswith("_")
        )
        file_list = [f.name for f in files]

        extracted = {
            "title": title,
            "client": client,
            "sector": sector,
            "country": country,
            "tender_number": tender_number,
            "technical_summary": technical_summary,
            "pricing_summary": pricing_summary,
            "total_price": total_price,
            "margin_info": margin_info,
            "technologies": techs,
            "keywords": kws,
            "full_summary": full_summary,
        }

        return await _save_extracted(
            folder_name, folder_path, extracted,
            len(files), file_list,
        )

    @mcp.tool()
    async def search_past_proposals(
        query: str, sector: str = "", limit: int = 5, mode: str = "auto"
    ) -> dict:
        """Search indexed past proposals using keyword, semantic, or hybrid search.

        Modes:
        - "auto": Uses hybrid (FTS5 + vector RRF) if embeddings are available, otherwise FTS5-only
        - "keyword": FTS5 only — supports quoted phrases ("core network"), prefix (cisco*), boolean (AND/OR)
        - "semantic": Vector similarity only — finds conceptually similar proposals even without exact keyword matches
        - "hybrid": Combines FTS5 + vector using Reciprocal Rank Fusion for best results

        Args:
            query: Search query text
            sector: Optional sector filter (telecom, it, infrastructure, security, energy, general)
            limit: Maximum results to return (default 5)
            mode: Search mode — "auto", "keyword", "semantic", or "hybrid"

        Returns:
            Dict with matches (ranked list), result_count, and search_mode used
        """
        use_fts = mode in ("auto", "keyword", "hybrid")
        use_vec = mode in ("auto", "semantic", "hybrid")
        has_vec = embeddings is not None and db.vec_enabled

        # Determine actual mode
        if mode == "auto":
            actual_mode = "hybrid" if has_vec else "keyword"
        elif mode in ("semantic", "hybrid") and not has_vec:
            actual_mode = "keyword"
            logger.info("Vector search unavailable, falling back to keyword mode")
        else:
            actual_mode = mode

        fts_results = []
        vec_results = []

        # FTS5 search
        if actual_mode in ("keyword", "hybrid"):
            try:
                fts_results = await db.search_proposal_index(
                    query, sector=sector, limit=limit * 2
                )
            except Exception as e:
                logger.warning("FTS5 search failed: %s", e)

        # Vector search
        if actual_mode in ("semantic", "hybrid") and has_vec:
            try:
                query_vec = await embeddings.embed_query(query)
                vec_results = await db.search_proposal_vector(
                    query_vec, limit=limit * 2
                )
                # Apply sector filter to vector results if specified
                if sector and vec_results:
                    vec_results = [
                        r for r in vec_results
                        if r.get("sector", "").lower() == sector.lower()
                    ]
            except Exception as e:
                logger.warning("Vector search failed: %s", e)

        # Combine results
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
