"""Document Intelligence tools — RFP parsing, compliance matrix, deadlines, validation.

Data-tool pattern: tools return raw data for Claude to analyze, or accept
Claude-structured data to store. No server-side LLM calls required.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from app.db.database import Database
from app.services.docwriter import DocWriterService
from app.services.parser import ParserService

logger = logging.getLogger(__name__)


def register_document_tools(
    mcp: FastMCP,
    db: Database,
    parser: ParserService,
    docwriter: DocWriterService,
    data_dir: Path,
) -> None:
    """Register all document intelligence tools on the MCP server."""

    @mcp.tool()
    async def parse_tender_rfp(file_path: str) -> dict:
        """Parse a tender RFP document (PDF or DOCX) and extract raw text content.

        Extracts all text and tables from the document. Returns the raw content
        for you (Claude) to analyze and structure. After analyzing, call save_rfp
        to store the structured data.

        Args:
            file_path: Path to the RFP document (PDF, DOCX, or XLSX)

        Returns:
            Dict with text, tables, page_count, and format
        """
        parsed = await parser.parse_file(file_path)

        # Copy file to rfp_documents directory
        rfp_docs_dir = data_dir / "rfp_documents"
        rfp_docs_dir.mkdir(parents=True, exist_ok=True)
        dest_path = rfp_docs_dir / Path(file_path).name
        if Path(file_path).resolve() != dest_path.resolve():
            shutil.copy2(file_path, dest_path)

        logger.info("Parsed RFP document: %s (%d chars)", file_path, len(parsed.get("text", "")))

        return {
            "text": parsed["text"],
            "tables": parsed.get("tables", []),
            "page_count": parsed.get("page_count", 0),
            "format": parsed.get("format", Path(file_path).suffix.lstrip(".")),
            "archived_path": str(dest_path),
        }

    @mcp.tool()
    async def save_rfp(
        title: str,
        client: str,
        sector: str = "telecom",
        country: str = "OM",
        rfp_number: str = "",
        deadline: str = "",
        submission_method: str = "",
        file_path: str = "",
        parsed_sections: Optional[dict] = None,
        requirements: Optional[list] = None,
        evaluation_criteria: Optional[list] = None,
        notes: str = "",
    ) -> dict:
        """Save structured RFP data to the database.

        Call this after you have analyzed the raw text from parse_tender_rfp
        and structured it into title, client, requirements, etc.

        Args:
            title: Full tender title
            client: Issuing organization name
            sector: Sector (telecom, it, infrastructure, security, general)
            country: Country code (default: OM)
            rfp_number: Reference number
            deadline: Submission deadline in YYYY-MM-DD format
            submission_method: How to submit (email, portal, physical)
            file_path: Path to the archived RFP file
            parsed_sections: Dict of {section_name: description}
            requirements: List of requirements extracted from the RFP
            evaluation_criteria: List of {criterion, weight} dicts
            notes: Additional notes

        Returns:
            The saved RFP record with its ID
        """
        rfp = await db.create_rfp(
            title=title,
            client=client,
            sector=sector,
            country=country,
            rfp_number=rfp_number or None,
            deadline=deadline or None,
            submission_method=submission_method or None,
            status="analyzing",
            file_path=file_path,
            parsed_sections=parsed_sections or {},
            requirements=requirements or [],
            evaluation_criteria=evaluation_criteria or [],
            notes=notes,
        )

        logger.info("Saved RFP: %s (id=%s)", rfp["title"], rfp["id"])
        return rfp

    @mcp.tool()
    async def get_rfp(rfp_id: str) -> dict:
        """Fetch an RFP record from the database.

        Args:
            rfp_id: ID of the RFP

        Returns:
            Full RFP record including parsed sections, requirements, and evaluation criteria
        """
        rfp = await db.get_rfp(rfp_id)
        if not rfp:
            raise ValueError(f"RFP not found: {rfp_id}")
        return rfp

    @mcp.tool()
    async def list_rfps(status: str = "") -> list:
        """List all RFPs in the database, optionally filtered by status.

        Args:
            status: Optional filter (new, analyzing, in_progress, submitted, awarded, lost, cancelled)

        Returns:
            List of RFP records
        """
        return await db.list_rfps(status=status)

    @mcp.tool()
    async def export_compliance_matrix(rfp_id: str, responses: list[dict]) -> str:
        """Generate a compliance matrix DOCX from your analysis.

        You (Claude) should analyze each requirement and provide the responses.
        Each response dict should have: requirement, status, narrative.

        Args:
            rfp_id: ID of the parsed RFP
            responses: List of dicts, each with "requirement" (str),
                      "status" (str, e.g. "Compliant"), and "narrative" (str)

        Returns:
            File path to the generated compliance matrix DOCX
        """
        rfp = await db.get_rfp(rfp_id)
        if not rfp:
            raise ValueError(f"RFP not found: {rfp_id}")

        requirements = rfp.get("requirements", [])
        if not requirements and not responses:
            raise ValueError(f"No requirements found for RFP {rfp_id}.")

        # Use provided requirements list for the matrix structure
        req_dicts = [
            {"requirement": r["requirement"] if isinstance(r, dict) else r}
            for r in (responses if responses else requirements)
        ]

        output_path = docwriter.create_compliance_matrix(req_dicts, responses)
        logger.info("Generated compliance matrix: %s", output_path)
        return output_path

    @mcp.tool()
    async def check_submission_deadline(rfp_id: str) -> dict:
        """Check the submission deadline for an RFP and calculate time remaining.

        Args:
            rfp_id: ID of the parsed RFP

        Returns:
            Dict with deadline, days_remaining, status, and milestones
        """
        rfp = await db.get_rfp(rfp_id)
        if not rfp:
            raise ValueError(f"RFP not found: {rfp_id}")

        deadline_str = rfp.get("deadline")
        if not deadline_str:
            return {
                "deadline": None,
                "days_remaining": None,
                "status": "unknown",
                "message": "No deadline set for this RFP.",
                "milestones": [],
            }

        try:
            deadline = datetime.strptime(deadline_str, "%Y-%m-%d")
        except ValueError:
            return {
                "deadline": deadline_str,
                "days_remaining": None,
                "status": "unparseable",
                "message": f"Could not parse deadline format: {deadline_str}",
                "milestones": [],
            }

        now = datetime.now()
        days_remaining = (deadline - now).days

        if days_remaining < 0:
            status = "overdue"
        elif days_remaining <= 1:
            status = "critical"
        elif days_remaining <= 3:
            status = "urgent"
        elif days_remaining <= 7:
            status = "warning"
        elif days_remaining <= 14:
            status = "attention"
        else:
            status = "on_track"

        milestones = []
        milestone_defs = [
            (14, "Start proposal drafting"),
            (10, "Complete technical approach"),
            (7, "Internal review deadline"),
            (5, "Partner inputs due"),
            (3, "Final review and formatting"),
            (1, "Submission preparation"),
            (0, "Submission deadline"),
        ]
        for days_before, label in milestone_defs:
            m_date = deadline - timedelta(days=days_before)
            is_past = m_date < now
            milestones.append({
                "date": m_date.strftime("%Y-%m-%d"),
                "label": label,
                "days_before_deadline": days_before,
                "completed": is_past,
            })

        return {
            "rfp_title": rfp["title"],
            "deadline": deadline_str,
            "days_remaining": days_remaining,
            "status": status,
            "milestones": milestones,
        }

    @mcp.tool()
    async def validate_document_completeness(rfp_id: str) -> dict:
        """Validate that a proposal has all required sections.

        Args:
            rfp_id: ID of the parsed RFP

        Returns:
            Dict with complete (bool), missing_sections, warnings, and section_status
        """
        rfp = await db.get_rfp(rfp_id)
        if not rfp:
            raise ValueError(f"RFP not found: {rfp_id}")

        proposals = await db.get_proposals_for_rfp(rfp_id)

        mandatory_sections = [
            "Executive Summary",
            "Technical Approach",
            "Solution Architecture",
            "Implementation Methodology",
            "Project Timeline",
            "Team Qualifications",
            "Past Experience",
        ]

        existing_sections = set()
        if proposals:
            if isinstance(proposals, list):
                for prop in proposals:
                    for sec in prop.get("sections", []):
                        existing_sections.add(sec.get("title", "").lower())
            elif isinstance(proposals, dict):
                for sec in proposals.get("sections", []):
                    existing_sections.add(sec.get("title", "").lower())

        section_status = []
        missing = []
        for section in mandatory_sections:
            found = section.lower() in existing_sections
            section_status.append({"section": section, "present": found})
            if not found:
                missing.append(section)

        warnings = []
        if not rfp.get("deadline"):
            warnings.append("No submission deadline set — risk of missing submission window.")
        if not rfp.get("requirements"):
            warnings.append("No requirements extracted from RFP — compliance matrix will be empty.")
        if not proposals:
            warnings.append("No proposal documents created yet.")

        complete = len(missing) == 0 and not any("No proposal" in w for w in warnings)

        return {
            "rfp_title": rfp["title"],
            "complete": complete,
            "missing_sections": missing,
            "section_status": section_status,
            "warnings": warnings,
            "total_sections": len(mandatory_sections),
            "completed_sections": len(mandatory_sections) - len(missing),
        }
