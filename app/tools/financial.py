"""Financial Proposal tools — vendor quotes, BOM, pricing, financial proposal generation.

Data-tool pattern: ingest_vendor_quote returns raw text for Claude to analyze.
Claude extracts items and calls save_vendor_items to store them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from app.db.database import Database
from app.services.docwriter import DocWriterService
from app.services.parser import ParserService

logger = logging.getLogger(__name__)


def register_financial_tools(
    mcp: FastMCP,
    db: Database,
    parser: ParserService,
    docwriter: DocWriterService,
    data_dir: Path,
    default_currency: str,
    default_margin_pct: float,
) -> None:
    """Register all financial proposal tools on the MCP server."""

    @mcp.tool()
    async def ingest_vendor_quote(vendor_name: str, quote_file: str) -> dict:
        """Parse a vendor quote document and return raw extracted content.

        Returns the raw text and tables from the quote file. You (Claude) should
        analyze the content, extract line items, then call save_vendor_items
        to store the structured data.

        Args:
            vendor_name: Name of the vendor (e.g., "Cisco", "Palo Alto Networks")
            quote_file: Path to the quote document (PDF or XLSX)

        Returns:
            Dict with vendor_name, raw text, tables, and format
        """
        parsed = await parser.parse_file(quote_file)

        # Copy to vendor_quotes directory
        quotes_dir = data_dir / "vendor_quotes"
        quotes_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Parsed vendor quote: %s (%d chars)", vendor_name, len(parsed.get("text", "")))

        return {
            "vendor_name": vendor_name,
            "text": parsed["text"],
            "tables": parsed.get("tables", []),
            "page_count": parsed.get("page_count", 0),
            "format": parsed.get("format", ""),
        }

    @mcp.tool()
    async def save_vendor_items(
        vendor_name: str,
        currency: str,
        items: list[dict],
    ) -> dict:
        """Save structured vendor quote items that you (Claude) extracted.

        Args:
            vendor_name: Name of the vendor
            currency: Currency code (e.g., "USD", "OMR", "EUR")
            items: List of dicts, each with: category, item_name, description,
                  manufacturer, part_number, quantity, unit, unit_cost

        Returns:
            Dict with vendor_id, items_saved, and total
        """
        vendor = await db.upsert_vendor(
            name=vendor_name,
            currency=currency or default_currency,
        )

        total = sum(
            item.get("quantity", 1) * item.get("unit_cost", 0) for item in items
        )

        logger.info(
            "Saved vendor items: %s (%d items, total=%.2f %s)",
            vendor_name, len(items), total, currency,
        )

        return {
            "vendor_id": vendor["id"],
            "vendor_name": vendor_name,
            "items_saved": len(items),
            "total": total,
            "currency": currency or default_currency,
            "items": items,
        }

    @mcp.tool()
    async def build_bom(rfp_id: str, vendor_quotes: list[dict]) -> dict:
        """Build a Bill of Materials from multiple vendor quotes.

        Args:
            rfp_id: ID of the parsed RFP
            vendor_quotes: List of dicts, each with "vendor_name" and "items"
                          (as structured by save_vendor_items)

        Returns:
            Dict with proposal_id, item_count, subtotal, and by_category breakdown
        """
        rfp = await db.get_rfp(rfp_id)
        if not rfp:
            raise ValueError(f"RFP not found: {rfp_id}")

        proposal = await db.create_proposal(
            rfp_id=rfp_id,
            proposal_type="financial",
            title=f"Financial Proposal — {rfp['title']}",
        )

        item_count = 0
        sort_order = 0

        for quote in vendor_quotes:
            vendor_name = quote.get("vendor_name", "Unknown")
            vendor = await db.get_vendor_by_name(vendor_name)
            vendor_id = vendor["id"] if vendor else None

            for item in quote.get("items", []):
                await db.add_bom_item(
                    proposal_id=proposal["id"],
                    category=item.get("category", "general"),
                    item_name=item.get("item_name", "Unknown Item"),
                    unit_cost=float(item.get("unit_cost", 0)),
                    description=item.get("description", ""),
                    vendor_id=vendor_id,
                    manufacturer=item.get("manufacturer", vendor_name),
                    part_number=item.get("part_number", ""),
                    quantity=float(item.get("quantity", 1)),
                    unit=item.get("unit", "unit"),
                    margin_pct=default_margin_pct,
                    warranty_months=item.get("warranty_months", 12),
                    sort_order=sort_order,
                )
                item_count += 1
                sort_order += 1

        totals = await db.get_bom_totals(proposal["id"])

        logger.info("Built BOM for RFP %s: %d items, total=%.2f", rfp_id, item_count, totals["total"])

        return {
            "proposal_id": proposal["id"],
            "item_count": item_count,
            "subtotal": totals["total"],
            "by_category": totals["by_category"],
            "currency": default_currency,
        }

    @mcp.tool()
    async def calculate_final_pricing(
        proposal_id: str, margin_rules: dict | None = None
    ) -> dict:
        """Calculate final pricing with margin adjustments.

        Args:
            proposal_id: ID of the financial proposal
            margin_rules: Optional dict mapping category to margin %.
                         Example: {"hardware": 12, "software": 20, "services": 25}

        Returns:
            Dict with total, by_category breakdown, currency, and item_count
        """
        proposal = await db.get_proposal(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")

        bom_items = await db.get_bom_for_proposal(proposal_id)
        if not bom_items:
            raise ValueError(f"No BOM items found for proposal {proposal_id}")

        if margin_rules:
            for item in bom_items:
                category = item.get("category", "").lower()
                if category in margin_rules:
                    new_margin = margin_rules[category]
                    if item["margin_pct"] != new_margin:
                        await db.update_bom_item(item["id"], margin_pct=new_margin)

        totals = await db.get_bom_totals(proposal_id)

        logger.info("Calculated pricing for proposal %s: total=%.2f", proposal_id, totals["total"])

        return {
            "proposal_id": proposal_id,
            "total": totals["total"],
            "by_category": totals["by_category"],
            "item_count": totals["item_count"],
            "currency": default_currency,
            "margin_rules_applied": margin_rules or {"default": default_margin_pct},
        }

    @mcp.tool()
    async def generate_financial_proposal(rfp_id: str, proposal_id: str) -> str:
        """Generate financial proposal DOCX + BOM XLSX from database data.

        Args:
            rfp_id: ID of the parsed RFP
            proposal_id: ID of the financial proposal (with BOM items)

        Returns:
            File path to the generated financial proposal DOCX
        """
        rfp = await db.get_rfp(rfp_id)
        if not rfp:
            raise ValueError(f"RFP not found: {rfp_id}")

        proposal = await db.get_proposal(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")

        bom_items = await db.get_bom_for_proposal(proposal_id)
        if not bom_items:
            raise ValueError(f"No BOM items found for proposal {proposal_id}")

        metadata = {
            "client": rfp["client"],
            "company": "TenderAI",
            "rfp_number": rfp.get("rfp_number", ""),
            "rfp_id": rfp_id,
            "title": rfp["title"],
            "currency": default_currency,
        }

        docx_path = docwriter.create_financial_proposal(bom_items, metadata)
        xlsx_path = docwriter.create_bom_spreadsheet(bom_items, metadata)

        await db.update_proposal(proposal_id, output_path=docx_path, status="review")

        logger.info("Generated financial proposal: %s (BOM: %s)", docx_path, xlsx_path)
        return docx_path
