# TenderAI — Project Instructions for Claude Code

## What This Is

TenderAI is a FastMCP server for tender/proposal management. It provides **data-only MCP tools** — Claude (via Max/Pro subscription) does all the reasoning, analysis, and writing. No `ANTHROPIC_API_KEY` required.

## Architecture: Data-Tool Pattern

Tools follow a parse → analyze → save pattern:
1. **Parse tools** return raw text/data (e.g., `parse_tender_rfp`, `ingest_vendor_quote`, `index_past_proposal`)
2. **Claude analyzes** the raw data and structures it
3. **Save tools** store the structured data (e.g., `save_rfp`, `save_vendor_items`, `save_proposal_index`)
4. **Context tools** load grounding data for Claude to use (e.g., `get_proposal_context`, `get_partner_brief_context`)
5. **Assembly tools** generate documents from saved data (e.g., `assemble_technical_proposal`, `generate_financial_proposal`)

## Running the Server

```bash
# Local (stdio)
cd /home/kitchen/Desktop/tenders
source venv/bin/activate
python -m app.server

# HTTP transport
TRANSPORT=http python -m app.server
```

## Project Structure

```
app/
├── server.py              # Entry point — wires all dependencies
├── config.py              # Settings from .env (dotenv)
├── tools/                 # MCP tools (5 modules, 22 tools, all data-only)
│   ├── document.py        # 7 tools: parse/save RFP, get/list RFPs, compliance, deadline, validation
│   ├── technical.py       # 3 tools: get context, save section, assemble proposal
│   ├── financial.py       # 5 tools: ingest/save vendor items, BOM, pricing, financial proposal
│   ├── partners.py        # 3 tools: partner context, NDA checklist, deliverable tracking
│   └── indexing.py        # 4 tools: index/save proposal metadata, search (hybrid), list indexed
├── services/
│   ├── llm.py             # Anthropic SDK wrapper (optional — not used in data-tool mode)
│   ├── parser.py          # PDF/DOCX/XLSX parser (pdfplumber, python-docx, openpyxl)
│   ├── embeddings.py      # Voyage AI async wrapper for vector embeddings
│   └── docwriter.py       # DOCX/XLSX document generation
├── db/
│   ├── schema.sql         # 7 tables + FTS5 virtual table + 3 sync triggers
│   ├── database.py        # Async SQLite layer + sqlite-vec extension + vector methods
│   └── models.py          # Pydantic models for all entities
├── resources/
│   └── knowledge.py       # 5 MCP resource URI handlers
├── prompts/
│   └── workflows.py       # 4 workflow prompts
└── middleware/
    └── auth.py            # ASGI Bearer token auth middleware
```

## Database

- SQLite at `db/tenderai.db` (WAL mode, auto-created on startup)
- 7 tables: rfp, proposal, vendor, bom, partner, partner_deliverable, past_proposal_index
- FTS5 virtual table: `past_proposal_fts` (keyword search with BM25 + porter stemming)
- vec0 virtual table: `past_proposal_vec` (vector similarity search via sqlite-vec)
- Schema runs on every startup via `CREATE TABLE IF NOT EXISTS` — no migrations needed

## Search System

Past proposals are indexed for fast retrieval:

1. **FTS5 (always available)**: Keyword search with BM25 ranking, porter stemming, unicode61 tokenizer
2. **Vector search (optional)**: sqlite-vec + Voyage AI embeddings (voyage-3-lite, 512 dims)
3. **Hybrid search**: Reciprocal Rank Fusion (RRF, k=60) combining both

Cascade fallback in `get_proposal_context()`: vector search → FTS5 → `_summary.md` files → raw filesystem scan

Set `VOYAGE_API_KEY` in `.env` to enable vector search. Without it, FTS5 keyword search is still fully functional.

## Key Conventions

- All DB IDs are 12-char hex UUIDs (`uuid.uuid4().hex[:12]`)
- JSON fields (lists/dicts) are stored as TEXT, serialized/deserialized in database.py
- Tool registration follows the pattern: `register_X_tools(mcp, db, ...)` in each module
- Each tool module's registration function is a closure that captures shared dependencies
- `LLMService` is fully optional — created only if `ANTHROPIC_API_KEY` is set
- Past proposal files go in `data/past_proposals/{folder_name}/`
- Generated output goes to `data/generated_proposals/`

## Configuration

All settings via `.env` (see `.env.example`). Key vars:
- `ANTHROPIC_API_KEY` — **optional**, not needed with Claude Max/Pro
- `VOYAGE_API_KEY` — optional, enables vector search for past proposals
- `TRANSPORT` — `stdio` (default) or `http`
- `MCP_API_KEY` — required for HTTP transport auth
- `COMPANY_NAME` — used in proposal generation

## Dependencies

Core: mcp[cli], aiosqlite, pdfplumber, python-docx, openpyxl, pydantic, python-dotenv, uvicorn, httpx
Optional: anthropic (only if using server-side AI), sqlite-vec, voyageai

## Testing

```bash
source venv/bin/activate

# Quick import check
python -c "from app.server import build_server; print('OK')"

# Full server build + DB connect
python -c "
import asyncio
from app.config import load_settings
from app.server import build_server
settings = load_settings()
mcp, db = build_server(settings)
asyncio.run(db.connect())
print(f'Tools: {len(mcp._tool_manager._tools)}')
print(f'Vec enabled: {db.vec_enabled}')
"
```
