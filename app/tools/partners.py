"""Partner Coordination tools — briefs, NDA checklists, deliverable tracking.

Data-tool pattern: get_partner_brief_context returns context for Claude to write the brief.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from app.db.database import Database

logger = logging.getLogger(__name__)


def register_partner_tools(
    mcp: FastMCP,
    db: Database,
    data_dir: Path,
) -> None:
    """Register all partner coordination tools on the MCP server."""

    @mcp.tool()
    async def get_partner_brief_context(partner_name: str, rfp_id: str) -> dict:
        """Load context for drafting a partner/subcontractor technical brief.

        Returns the RFP requirements, partner profile, and brief structure guidelines.
        You (Claude) should write the brief using this context.

        Args:
            partner_name: Name of the partner company
            rfp_id: ID of the parsed RFP

        Returns:
            Dict with rfp_data, partner_profile, and brief_structure guidelines
        """
        rfp = await db.get_rfp(rfp_id)
        if not rfp:
            raise ValueError(f"RFP not found: {rfp_id}")

        # Ensure partner exists in DB
        partner = await db.get_partner_by_name(partner_name)
        if not partner:
            partner = await db.upsert_partner(name=partner_name)

        return {
            "rfp_data": {
                "title": rfp["title"],
                "client": rfp["client"],
                "sector": rfp["sector"],
                "country": rfp["country"],
                "deadline": rfp.get("deadline", "TBD"),
                "requirements": rfp.get("requirements", []),
            },
            "partner_profile": {
                "id": partner["id"],
                "name": partner["name"],
                "country": partner.get("country", ""),
                "specialization": partner.get("specialization", ""),
                "nda_status": partner.get("nda_status", "none"),
                "past_projects": partner.get("past_projects", []),
                "notes": partner.get("notes", ""),
            },
            "brief_structure": [
                "1. Project Background — brief overview of the tender and client",
                "2. Scope of Work — specific deliverables required from the partner",
                "3. Technical Requirements — specifications and standards to meet",
                "4. Deliverables — list of documents/items expected",
                "5. Timeline — key dates and deadlines",
                "6. Submission Format — how to submit their input",
                "7. Confidentiality — note about NDA requirements",
            ],
        }

    @mcp.tool()
    async def create_nda_checklist(partner_name: str, rfp_id: str) -> dict:
        """Create an NDA checklist for a partner engagement.

        Args:
            partner_name: Name of the partner company
            rfp_id: ID of the parsed RFP

        Returns:
            Dict with checklist_items list and partner_id
        """
        rfp = await db.get_rfp(rfp_id)
        if not rfp:
            raise ValueError(f"RFP not found: {rfp_id}")

        partner = await db.get_partner_by_name(partner_name)
        if not partner:
            partner = await db.upsert_partner(name=partner_name)

        checklist_items = [
            {
                "item": "Confidentiality Scope",
                "description": f"Define what information related to '{rfp['title']}' is considered confidential, including RFP documents, pricing, technical designs, and client information.",
                "status": "pending",
            },
            {
                "item": "Term and Duration",
                "description": "NDA should be effective from signing date and remain in force for a minimum of 3 years after project completion or termination of discussions.",
                "status": "pending",
            },
            {
                "item": "Permitted Disclosures",
                "description": "Specify that confidential information may only be shared with employees and subcontractors who need-to-know, and who are bound by similar obligations.",
                "status": "pending",
            },
            {
                "item": "Jurisdiction and Governing Law",
                "description": f"NDA governed by laws of {rfp.get('country', 'OM')}. Disputes to be resolved through arbitration in the agreed jurisdiction.",
                "status": "pending",
            },
            {
                "item": "Return/Destroy Obligations",
                "description": "Upon termination or request, all confidential materials must be returned or destroyed, with written confirmation provided within 30 days.",
                "status": "pending",
            },
            {
                "item": "Exceptions to Confidentiality",
                "description": "Standard carve-outs: publicly available information, independently developed information, information received from third parties without restriction.",
                "status": "pending",
            },
            {
                "item": "Non-Solicitation",
                "description": "Neither party shall solicit or hire employees of the other party for the duration of the NDA plus 12 months.",
                "status": "pending",
            },
            {
                "item": "Breach Remedies",
                "description": "Define remedies for breach including injunctive relief and indemnification for damages caused by unauthorized disclosure.",
                "status": "pending",
            },
        ]

        await db.update_partner(partner["id"], nda_status="sent")

        logger.info("Created NDA checklist for %s (%d items)", partner_name, len(checklist_items))

        return {
            "partner_id": partner["id"],
            "partner_name": partner_name,
            "rfp_title": rfp["title"],
            "checklist_items": checklist_items,
            "nda_status": "sent",
        }

    @mcp.tool()
    async def track_partner_deliverable(
        partner_name: str, rfp_id: str, item: str, deadline: str
    ) -> dict:
        """Track a deliverable expected from a partner.

        Args:
            partner_name: Name of the partner company
            rfp_id: ID of the parsed RFP
            item: Description of the deliverable
            deadline: Due date in YYYY-MM-DD format

        Returns:
            Dict with deliverable_id, status, and tracking details
        """
        rfp = await db.get_rfp(rfp_id)
        if not rfp:
            raise ValueError(f"RFP not found: {rfp_id}")

        partner = await db.get_partner_by_name(partner_name)
        if not partner:
            partner = await db.upsert_partner(name=partner_name)

        proposals = await db.get_proposals_for_rfp(rfp_id)
        if isinstance(proposals, list) and proposals:
            proposal_id = proposals[0]["id"]
        elif isinstance(proposals, dict):
            proposal_id = proposals["id"]
        else:
            prop = await db.create_proposal(
                rfp_id=rfp_id,
                proposal_type="technical",
                title=f"Proposal — {rfp['title']}",
            )
            proposal_id = prop["id"]

        item_lower = item.lower()
        if any(kw in item_lower for kw in ("price", "pricing", "cost", "quote")):
            deliv_type = "pricing"
        elif any(kw in item_lower for kw in ("cv", "resume", "personnel")):
            deliv_type = "cv"
        elif any(kw in item_lower for kw in ("cert", "certificate", "accreditation")):
            deliv_type = "certification"
        elif any(kw in item_lower for kw in ("reference", "letter")):
            deliv_type = "reference_letter"
        elif any(kw in item_lower for kw in ("technical", "spec", "design", "architecture")):
            deliv_type = "technical_input"
        else:
            deliv_type = "document"

        deliverable = await db.create_deliverable(
            partner_id=partner["id"],
            proposal_id=proposal_id,
            title=item,
            deliverable_type=deliv_type,
            due_date=deadline,
            status="requested",
        )

        logger.info(
            "Tracking deliverable from %s: '%s' (due: %s)",
            partner_name, item, deadline,
        )

        return {
            "deliverable_id": deliverable["id"],
            "partner_name": partner_name,
            "partner_id": partner["id"],
            "item": item,
            "deliverable_type": deliv_type,
            "deadline": deadline,
            "status": "requested",
            "proposal_id": proposal_id,
        }
