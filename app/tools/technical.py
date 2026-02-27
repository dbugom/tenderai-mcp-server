"""Technical Proposal tools — context loading, section saving, proposal assembly.

Data-tool pattern: tools provide context and store results. Claude does the writing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from app.db.database import Database
from app.services.docwriter import DocWriterService
from app.services.parser import ParserService

logger = logging.getLogger(__name__)

# Supported file extensions for past proposals
PAST_PROPOSAL_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".md", ".txt"}

DEFAULT_SECTIONS = [
    "Company Profile",
    "Past Successful Projects",
    "Executive Summary",
    "Technical Approach",
    "Solution Architecture",
    "Implementation Methodology",
    "Project Timeline",
    "Team Qualifications",
    "Past Experience",
]


def register_technical_tools(
    mcp: FastMCP,
    db: Database,
    parser: ParserService,
    docwriter: DocWriterService,
    data_dir: Path,
    company_name: str,
    embeddings=None,
) -> None:
    """Register all technical proposal tools on the MCP server."""

    async def _load_company_profile() -> str:
        """Load company profile from knowledge base if available."""
        profile_dir = data_dir / "knowledge_base" / "company_profile"
        if not profile_dir.exists():
            return f"Company: {company_name}"

        md_path = profile_dir / "profile.md"
        if md_path.exists():
            return md_path.read_text()

        for ext in (".pdf", ".docx", ".doc"):
            for f in profile_dir.iterdir():
                if f.suffix.lower() == ext:
                    try:
                        parsed = await parser.parse_file(str(f))
                        return parsed["text"]
                    except Exception as e:
                        logger.warning("Could not parse company profile %s: %s", f, e)

        return f"Company: {company_name}"

    async def _find_past_proposals(rfp_id: str) -> list[dict]:
        """Find relevant past proposals using vector → FTS5 → filesystem cascade."""
        rfp = await db.get_rfp(rfp_id)
        search_terms = []
        if rfp:
            if rfp.get("title"):
                search_terms.append(rfp["title"])
            if rfp.get("sector"):
                search_terms.append(rfp["sector"])
            if rfp.get("client"):
                search_terms.append(rfp["client"])
        search_query = " ".join(search_terms)

        if search_query:
            # Try vector search first
            matches = []
            if embeddings and db.vec_enabled:
                try:
                    query_vec = await embeddings.embed_query(search_query)
                    matches = await db.search_proposal_vector(query_vec, limit=3)
                    if matches:
                        logger.debug("Found %d past proposals via vector search", len(matches))
                except Exception as e:
                    logger.debug("Vector search failed: %s", e)

            # Fall back to FTS5
            if not matches:
                try:
                    matches = await db.search_proposal_index(search_query, limit=3)
                    if matches:
                        logger.debug("Found %d past proposals via FTS5", len(matches))
                except Exception as e:
                    logger.debug("FTS5 search failed: %s", e)

            if matches:
                return matches

        return []

    async def _load_past_proposals_filesystem(section_name: str) -> list[str]:
        """Fallback: load past proposals from filesystem."""
        docs = []
        past_dir = data_dir / "past_proposals"
        section_key = section_name.lower().replace(" ", "_")
        if not past_dir.exists():
            return docs

        for proposal_dir in sorted(past_dir.iterdir()):
            if not proposal_dir.is_dir():
                continue
            summary_file = proposal_dir / "_summary.md"
            if summary_file.exists():
                content = summary_file.read_text()
                docs.append(f"Past proposal reference ({proposal_dir.name}):\n{content[:3000]}")
                continue

            for f in sorted(proposal_dir.iterdir()):
                if f.suffix.lower() in PAST_PROPOSAL_EXTENSIONS and section_key in f.name.lower():
                    ext = f.suffix.lower()
                    if ext in (".md", ".txt"):
                        content = f.read_text()
                    elif ext in (".pdf", ".docx", ".doc", ".xlsx", ".xls"):
                        try:
                            parsed = await parser.parse_file(str(f))
                            content = parsed["text"]
                        except Exception:
                            content = ""
                    else:
                        content = ""
                    if content:
                        docs.append(f"Past proposal reference ({proposal_dir.name}/{f.name}):\n{content[:3000]}")
                    break

        return docs

    @mcp.tool()
    async def get_proposal_context(rfp_id: str, section_name: str = "") -> dict:
        """Load all grounding context for writing a proposal section.

        Returns the company profile, RFP data, relevant templates, and past proposal
        references. Use this context to write the section, then call save_proposal_section
        to store your written content.

        Args:
            rfp_id: ID of the parsed RFP
            section_name: Section being written (e.g., "Technical Approach").
                         Helps load the most relevant past proposal references.

        Returns:
            Dict with company_profile, rfp_data, template (if available),
            past_proposal_references, and section_guidelines
        """
        rfp = await db.get_rfp(rfp_id)
        if not rfp:
            raise ValueError(f"RFP not found: {rfp_id}")

        # Company profile
        company_profile = await _load_company_profile()

        # RFP context
        rfp_data = {
            "title": rfp["title"],
            "client": rfp["client"],
            "sector": rfp["sector"],
            "country": rfp["country"],
            "rfp_number": rfp.get("rfp_number", ""),
            "deadline": rfp.get("deadline", ""),
            "requirements": rfp.get("requirements", []),
            "parsed_sections": rfp.get("parsed_sections", {}),
            "evaluation_criteria": rfp.get("evaluation_criteria", []),
        }

        # Template
        template = None
        if section_name:
            template_path = data_dir / "knowledge_base" / "templates" / f"{section_name.lower().replace(' ', '_')}.md"
            if template_path.exists():
                template = template_path.read_text()

        # Past proposals
        past_refs = []
        matches = await _find_past_proposals(rfp_id)
        if matches:
            for m in matches:
                techs = ", ".join(m.get("technologies", [])[:10])
                past_refs.append({
                    "folder_name": m.get("folder_name", ""),
                    "title": m.get("title", ""),
                    "client": m.get("client", ""),
                    "sector": m.get("sector", ""),
                    "technologies": techs,
                    "technical_summary": m.get("technical_summary", "")[:3000],
                })
        else:
            # Filesystem fallback
            fs_docs = await _load_past_proposals_filesystem(section_name or "general")
            for doc in fs_docs[:3]:
                past_refs.append({"raw_content": doc})

        # Section guidelines
        section_guidelines = {
            "Executive Summary": "400-600 words. Address client by name, restate objectives, present solution, highlight differentiators, close with commitment.",
            "Technical Approach": "800-1200 words. Describe solution architecture, technology choices, and how they address each requirement.",
            "Solution Architecture": "800-1200 words. Cover topology, components, redundancy, security, integration points.",
            "Implementation Methodology": "600-1000 words. Phased approach, milestones, deliverables, quality gates, risk mitigation.",
            "Project Timeline": "400-800 words. Phase-by-phase timeline with durations, dependencies, milestones.",
            "Team Qualifications": "400-800 words. Team structure, key personnel, certifications, experience.",
            "Past Experience": "400-800 words. 3-5 relevant projects with scope, technologies, outcomes.",
            "Company Profile": "400-800 words. Legal name, establishment, certifications, employees, geographic presence.",
            "Past Successful Projects": "400-800 words. Relevant projects with scope, value, technologies, outcomes.",
        }

        return {
            "company_profile": company_profile,
            "rfp_data": rfp_data,
            "template": template,
            "past_proposal_references": past_refs,
            "section_guidelines": section_guidelines.get(section_name, "500-1000 words. Formal proposal language."),
        }

    @mcp.tool()
    async def save_proposal_section(
        rfp_id: str, section_name: str, content: str
    ) -> dict:
        """Save a proposal section that you (Claude) have written.

        Args:
            rfp_id: ID of the parsed RFP
            section_name: Name of the section (e.g., "Executive Summary", "Technical Approach")
            content: The section content you wrote

        Returns:
            Dict with section_name, word_count, and proposal_id
        """
        rfp = await db.get_rfp(rfp_id)
        if not rfp:
            raise ValueError(f"RFP not found: {rfp_id}")

        # Find or create technical proposal
        proposals = await db.get_proposals_for_rfp(rfp_id, "technical")
        if isinstance(proposals, dict):
            proposal = proposals
        elif isinstance(proposals, list) and proposals:
            proposal = proposals[0]
        else:
            proposal = await db.create_proposal(
                rfp_id=rfp_id,
                proposal_type="technical",
                title=f"Technical Proposal — {rfp['title']}",
            )

        # Update sections
        sections = proposal.get("sections", [])
        section_id = section_name.lower().replace(" ", "_")

        updated = False
        for i, sec in enumerate(sections):
            if sec.get("section_id") == section_id:
                sections[i] = {
                    "section_id": section_id,
                    "title": section_name,
                    "content": content.strip(),
                    "order": i,
                }
                updated = True
                break
        if not updated:
            sections.append({
                "section_id": section_id,
                "title": section_name,
                "content": content.strip(),
                "order": len(sections),
            })

        await db.update_proposal(proposal["id"], sections=sections)

        word_count = len(content.split())
        logger.info("Saved section '%s' for RFP %s (%d words)", section_name, rfp_id, word_count)

        return {
            "section_name": section_name,
            "word_count": word_count,
            "proposal_id": proposal["id"],
        }

    @mcp.tool()
    async def assemble_technical_proposal(
        rfp_id: str, sections: list[str] | None = None
    ) -> str:
        """Assemble saved proposal sections into a formatted DOCX document.

        Uses sections already saved via save_proposal_section. The document follows
        standard tender structure: cover page, Company Profile, Past Successful
        Projects, TOC, then the technical body sections.

        Args:
            rfp_id: ID of the parsed RFP
            sections: Optional list of section names to include. Defaults to all 9 standard sections.

        Returns:
            File path to the generated DOCX document
        """
        rfp = await db.get_rfp(rfp_id)
        if not rfp:
            raise ValueError(f"RFP not found: {rfp_id}")

        section_list = sections or DEFAULT_SECTIONS

        # Get existing proposal with saved sections
        proposals = await db.get_proposals_for_rfp(rfp_id, "technical")
        if isinstance(proposals, dict):
            proposal = proposals
        elif isinstance(proposals, list) and proposals:
            proposal = proposals[0]
        else:
            raise ValueError(f"No technical proposal found for RFP {rfp_id}. Save sections first with save_proposal_section.")

        saved_sections = {
            sec.get("title", ""): sec.get("content", "")
            for sec in proposal.get("sections", [])
        }

        # Build document sections from saved content
        doc_sections = []
        missing = []
        for section_name in section_list:
            content = saved_sections.get(section_name, "")
            if content:
                doc_sections.append({
                    "title": section_name,
                    "content": content,
                })
            else:
                missing.append(section_name)

        if not doc_sections:
            raise ValueError(
                f"No sections saved yet. Use save_proposal_section to write sections first. "
                f"Missing: {', '.join(missing)}"
            )

        if missing:
            logger.warning("Assembling proposal with missing sections: %s", ", ".join(missing))

        metadata = {
            "client": rfp["client"],
            "company": company_name,
            "rfp_number": rfp.get("rfp_number", ""),
            "rfp_id": rfp_id,
        }
        output_path = docwriter.create_technical_proposal(
            title=f"Technical Proposal — {rfp['title']}",
            sections=doc_sections,
            metadata=metadata,
        )

        # Update proposal record
        await db.update_proposal(proposal["id"], output_path=output_path, status="review")

        logger.info("Assembled technical proposal: %s (%d sections)", output_path, len(doc_sections))
        return output_path
