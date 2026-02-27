"""Async SQLite database layer for TenderAI."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

import struct

import aiosqlite

logger = logging.getLogger(__name__)

# Default embedding dimensions — overridden by Settings at runtime
_DEFAULT_EMBEDDING_DIM = 512


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _serialize_vector(vec: list[float]) -> bytes:
    """Pack a list of floats into a compact binary blob for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


class Database:
    """Async wrapper around aiosqlite with CRUD helpers for every table."""

    def __init__(self, db_path: Path, embedding_dimensions: int = _DEFAULT_EMBEDDING_DIM):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._vec_enabled = False
        self._embedding_dim = embedding_dimensions

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._run_schema()
        await self._load_vec_extension()
        logger.info("Database connected: %s (vec=%s)", self.db_path, self._vec_enabled)

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def _run_schema(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        sql = schema_path.read_text()
        await self._db.executescript(sql)
        await self._db.commit()

    async def _load_vec_extension(self) -> None:
        """Load sqlite-vec extension and create vec0 virtual table if available."""
        try:
            import sqlite_vec
            await self._db.enable_load_extension(True)
            await self._db.load_extension(sqlite_vec.loadable_path())
            await self._db.enable_load_extension(False)

            # Create vec0 table (idempotent — IF NOT EXISTS)
            await self._db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS past_proposal_vec "
                f"USING vec0(embedding float[{self._embedding_dim}])"
            )
            await self._db.commit()
            self._vec_enabled = True
            logger.info("sqlite-vec loaded — vector search enabled (dim=%d)", self._embedding_dim)
        except ImportError:
            logger.info("sqlite-vec not installed — vector search disabled")
        except Exception as e:
            logger.warning("Could not load sqlite-vec: %s — vector search disabled", e)

    @property
    def vec_enabled(self) -> bool:
        return self._vec_enabled

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    async def _fetchone(self, sql: str, params: tuple = ()) -> Optional[dict]:
        cursor = await self._db.execute(sql, params)
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def _execute(self, sql: str, params: tuple = ()) -> None:
        await self._db.execute(sql, params)
        await self._db.commit()

    # ------------------------------------------------------------------
    # RFP
    # ------------------------------------------------------------------

    async def create_rfp(self, *, title: str, client: str, **kwargs) -> dict:
        rfp_id = _new_id()
        cols = ["id", "title", "client"]
        vals = [rfp_id, title, client]
        for key in (
            "sector", "country", "rfp_number", "issue_date", "deadline",
            "submission_method", "status", "file_path", "notes",
        ):
            if key in kwargs:
                cols.append(key)
                vals.append(kwargs[key])
        for json_key in ("parsed_sections", "requirements", "evaluation_criteria"):
            if json_key in kwargs:
                cols.append(json_key)
                vals.append(json.dumps(kwargs[json_key]))
        placeholders = ",".join("?" for _ in vals)
        col_str = ",".join(cols)
        await self._execute(
            f"INSERT INTO rfp ({col_str}) VALUES ({placeholders})", tuple(vals)
        )
        return await self.get_rfp(rfp_id)

    async def get_rfp(self, rfp_id: str) -> Optional[dict]:
        row = await self._fetchone("SELECT * FROM rfp WHERE id=?", (rfp_id,))
        if row:
            row["parsed_sections"] = json.loads(row.get("parsed_sections") or "{}")
            row["requirements"] = json.loads(row.get("requirements") or "[]")
            row["evaluation_criteria"] = json.loads(row.get("evaluation_criteria") or "[]")
        return row

    async def update_rfp(self, rfp_id: str, **kwargs) -> Optional[dict]:
        sets, vals = [], []
        for key, value in kwargs.items():
            if key in ("parsed_sections", "requirements", "evaluation_criteria"):
                value = json.dumps(value)
            sets.append(f"{key}=?")
            vals.append(value)
        sets.append("updated_at=datetime('now')")
        vals.append(rfp_id)
        await self._execute(
            f"UPDATE rfp SET {','.join(sets)} WHERE id=?", tuple(vals)
        )
        return await self.get_rfp(rfp_id)

    async def list_rfps(self, status: Optional[str] = None) -> list[dict]:
        if status:
            return await self._fetchall("SELECT * FROM rfp WHERE status=? ORDER BY created_at DESC", (status,))
        return await self._fetchall("SELECT * FROM rfp ORDER BY created_at DESC")

    # ------------------------------------------------------------------
    # Proposal
    # ------------------------------------------------------------------

    async def create_proposal(self, *, rfp_id: str, proposal_type: str, **kwargs) -> dict:
        proposal_id = _new_id()
        cols = ["id", "rfp_id", "proposal_type"]
        vals = [proposal_id, rfp_id, proposal_type]
        for key in ("status", "title", "output_path", "version"):
            if key in kwargs:
                cols.append(key)
                vals.append(kwargs[key])
        if "sections" in kwargs:
            cols.append("sections")
            vals.append(json.dumps(kwargs["sections"]))
        placeholders = ",".join("?" for _ in vals)
        col_str = ",".join(cols)
        await self._execute(
            f"INSERT INTO proposal ({col_str}) VALUES ({placeholders})", tuple(vals)
        )
        return await self.get_proposal(proposal_id)

    async def get_proposal(self, proposal_id: str) -> Optional[dict]:
        row = await self._fetchone("SELECT * FROM proposal WHERE id=?", (proposal_id,))
        if row:
            row["sections"] = json.loads(row.get("sections") or "[]")
        return row

    async def get_proposals_for_rfp(self, rfp_id: str, proposal_type: Optional[str] = None) -> list[dict]:
        if proposal_type:
            rows = await self._fetchall(
                "SELECT * FROM proposal WHERE rfp_id=? AND proposal_type=? ORDER BY version DESC",
                (rfp_id, proposal_type),
            )
        else:
            rows = await self._fetchall(
                "SELECT * FROM proposal WHERE rfp_id=? ORDER BY version DESC", (rfp_id,)
            )
        for row in rows:
            row["sections"] = json.loads(row.get("sections") or "[]")
        return rows

    async def update_proposal(self, proposal_id: str, **kwargs) -> Optional[dict]:
        sets, vals = [], []
        for key, value in kwargs.items():
            if key == "sections":
                value = json.dumps(value)
            sets.append(f"{key}=?")
            vals.append(value)
        sets.append("updated_at=datetime('now')")
        vals.append(proposal_id)
        await self._execute(
            f"UPDATE proposal SET {','.join(sets)} WHERE id=?", tuple(vals)
        )
        return await self.get_proposal(proposal_id)

    # ------------------------------------------------------------------
    # Vendor
    # ------------------------------------------------------------------

    async def upsert_vendor(self, *, name: str, **kwargs) -> dict:
        existing = await self._fetchone("SELECT * FROM vendor WHERE name=?", (name,))
        if existing:
            return await self.update_vendor(existing["id"], **kwargs)
        vendor_id = _new_id()
        cols = ["id", "name"]
        vals = [vendor_id, name]
        for key in (
            "category", "specialization", "country", "contact_name",
            "contact_email", "contact_phone", "currency", "notes",
            "is_approved", "rating",
        ):
            if key in kwargs:
                cols.append(key)
                vals.append(kwargs[key])
        if "past_projects" in kwargs:
            cols.append("past_projects")
            vals.append(json.dumps(kwargs["past_projects"]))
        placeholders = ",".join("?" for _ in vals)
        col_str = ",".join(cols)
        await self._execute(
            f"INSERT INTO vendor ({col_str}) VALUES ({placeholders})", tuple(vals)
        )
        return await self.get_vendor(vendor_id)

    async def get_vendor(self, vendor_id: str) -> Optional[dict]:
        row = await self._fetchone("SELECT * FROM vendor WHERE id=?", (vendor_id,))
        if row:
            row["past_projects"] = json.loads(row.get("past_projects") or "[]")
        return row

    async def get_vendor_by_name(self, name: str) -> Optional[dict]:
        row = await self._fetchone("SELECT * FROM vendor WHERE name=?", (name,))
        if row:
            row["past_projects"] = json.loads(row.get("past_projects") or "[]")
        return row

    async def update_vendor(self, vendor_id: str, **kwargs) -> Optional[dict]:
        sets, vals = [], []
        for key, value in kwargs.items():
            if key == "past_projects":
                value = json.dumps(value)
            sets.append(f"{key}=?")
            vals.append(value)
        sets.append("updated_at=datetime('now')")
        vals.append(vendor_id)
        await self._execute(
            f"UPDATE vendor SET {','.join(sets)} WHERE id=?", tuple(vals)
        )
        return await self.get_vendor(vendor_id)

    async def list_vendors(self) -> list[dict]:
        rows = await self._fetchall("SELECT * FROM vendor ORDER BY name")
        for row in rows:
            row["past_projects"] = json.loads(row.get("past_projects") or "[]")
        return rows

    # ------------------------------------------------------------------
    # BOM
    # ------------------------------------------------------------------

    async def add_bom_item(self, *, proposal_id: str, category: str, item_name: str, unit_cost: float, **kwargs) -> dict:
        item_id = _new_id()
        cols = ["id", "proposal_id", "category", "item_name", "unit_cost"]
        vals: list[Any] = [item_id, proposal_id, category, item_name, unit_cost]
        for key in (
            "description", "vendor_id", "manufacturer", "part_number",
            "quantity", "unit", "margin_pct", "warranty_months", "sort_order",
        ):
            if key in kwargs:
                cols.append(key)
                vals.append(kwargs[key])
        placeholders = ",".join("?" for _ in vals)
        col_str = ",".join(cols)
        await self._execute(
            f"INSERT INTO bom ({col_str}) VALUES ({placeholders})", tuple(vals)
        )
        return await self.get_bom_item(item_id)

    async def get_bom_item(self, item_id: str) -> Optional[dict]:
        return await self._fetchone("SELECT * FROM bom WHERE id=?", (item_id,))

    async def get_bom_for_proposal(self, proposal_id: str) -> list[dict]:
        return await self._fetchall(
            "SELECT * FROM bom WHERE proposal_id=? ORDER BY sort_order, category, item_name",
            (proposal_id,),
        )

    async def update_bom_item(self, item_id: str, **kwargs) -> Optional[dict]:
        sets, vals = [], []
        for key, value in kwargs.items():
            sets.append(f"{key}=?")
            vals.append(value)
        sets.append("updated_at=datetime('now')")
        vals.append(item_id)
        await self._execute(
            f"UPDATE bom SET {','.join(sets)} WHERE id=?", tuple(vals)
        )
        return await self.get_bom_item(item_id)

    async def get_bom_totals(self, proposal_id: str) -> dict:
        rows = await self._fetchall(
            "SELECT category, SUM(total_cost) as subtotal, COUNT(*) as item_count "
            "FROM bom WHERE proposal_id=? GROUP BY category ORDER BY category",
            (proposal_id,),
        )
        grand_total = sum(r["subtotal"] for r in rows)
        return {"by_category": rows, "total": grand_total, "item_count": sum(r["item_count"] for r in rows)}

    # ------------------------------------------------------------------
    # Partner
    # ------------------------------------------------------------------

    async def upsert_partner(self, *, name: str, **kwargs) -> dict:
        existing = await self._fetchone("SELECT * FROM partner WHERE name=?", (name,))
        if existing:
            return await self.update_partner(existing["id"], **kwargs)
        partner_id = _new_id()
        cols = ["id", "name"]
        vals = [partner_id, name]
        for key in (
            "country", "specialization", "contact_name", "contact_email",
            "contact_phone", "nda_status", "nda_signed_date", "nda_expiry_date", "notes",
        ):
            if key in kwargs:
                cols.append(key)
                vals.append(kwargs[key])
        if "past_projects" in kwargs:
            cols.append("past_projects")
            vals.append(json.dumps(kwargs["past_projects"]))
        placeholders = ",".join("?" for _ in vals)
        col_str = ",".join(cols)
        await self._execute(
            f"INSERT INTO partner ({col_str}) VALUES ({placeholders})", tuple(vals)
        )
        return await self.get_partner(partner_id)

    async def get_partner(self, partner_id: str) -> Optional[dict]:
        row = await self._fetchone("SELECT * FROM partner WHERE id=?", (partner_id,))
        if row:
            row["past_projects"] = json.loads(row.get("past_projects") or "[]")
        return row

    async def get_partner_by_name(self, name: str) -> Optional[dict]:
        row = await self._fetchone("SELECT * FROM partner WHERE name=?", (name,))
        if row:
            row["past_projects"] = json.loads(row.get("past_projects") or "[]")
        return row

    async def update_partner(self, partner_id: str, **kwargs) -> Optional[dict]:
        sets, vals = [], []
        for key, value in kwargs.items():
            if key == "past_projects":
                value = json.dumps(value)
            sets.append(f"{key}=?")
            vals.append(value)
        sets.append("updated_at=datetime('now')")
        vals.append(partner_id)
        await self._execute(
            f"UPDATE partner SET {','.join(sets)} WHERE id=?", tuple(vals)
        )
        return await self.get_partner(partner_id)

    async def list_partners(self) -> list[dict]:
        rows = await self._fetchall("SELECT * FROM partner ORDER BY name")
        for row in rows:
            row["past_projects"] = json.loads(row.get("past_projects") or "[]")
        return rows

    # ------------------------------------------------------------------
    # Partner Deliverables
    # ------------------------------------------------------------------

    async def create_deliverable(
        self, *, partner_id: str, proposal_id: str, title: str, **kwargs
    ) -> dict:
        deliv_id = _new_id()
        cols = ["id", "partner_id", "proposal_id", "title"]
        vals = [deliv_id, partner_id, proposal_id, title]
        for key in ("deliverable_type", "due_date", "status", "file_path", "notes"):
            if key in kwargs:
                cols.append(key)
                vals.append(kwargs[key])
        placeholders = ",".join("?" for _ in vals)
        col_str = ",".join(cols)
        await self._execute(
            f"INSERT INTO partner_deliverable ({col_str}) VALUES ({placeholders})",
            tuple(vals),
        )
        return await self.get_deliverable(deliv_id)

    async def get_deliverable(self, deliv_id: str) -> Optional[dict]:
        return await self._fetchone("SELECT * FROM partner_deliverable WHERE id=?", (deliv_id,))

    async def get_deliverables_for_proposal(self, proposal_id: str) -> list[dict]:
        return await self._fetchall(
            "SELECT pd.*, p.name as partner_name FROM partner_deliverable pd "
            "JOIN partner p ON pd.partner_id = p.id "
            "WHERE pd.proposal_id=? ORDER BY pd.due_date",
            (proposal_id,),
        )

    async def update_deliverable(self, deliv_id: str, **kwargs) -> Optional[dict]:
        sets, vals = [], []
        for key, value in kwargs.items():
            sets.append(f"{key}=?")
            vals.append(value)
        sets.append("updated_at=datetime('now')")
        vals.append(deliv_id)
        await self._execute(
            f"UPDATE partner_deliverable SET {','.join(sets)} WHERE id=?", tuple(vals)
        )
        return await self.get_deliverable(deliv_id)

    # ------------------------------------------------------------------
    # Past Proposal Index
    # ------------------------------------------------------------------

    _INDEX_JSON_FIELDS = ("technologies", "keywords", "file_list")

    def _deserialize_index(self, row: Optional[dict]) -> Optional[dict]:
        if row:
            for key in self._INDEX_JSON_FIELDS:
                row[key] = json.loads(row.get(key) or "[]")
        return row

    async def upsert_proposal_index(self, *, folder_name: str, **kwargs) -> dict:
        existing = await self._fetchone(
            "SELECT * FROM past_proposal_index WHERE folder_name=?", (folder_name,)
        )
        if existing:
            return await self.update_proposal_index(existing["id"], **kwargs)
        index_id = _new_id()
        cols = ["id", "folder_name"]
        vals: list[Any] = [index_id, folder_name]
        for key in (
            "tender_number", "title", "client", "sector", "country",
            "technical_summary", "pricing_summary", "total_price",
            "margin_info", "full_summary", "file_count",
        ):
            if key in kwargs:
                cols.append(key)
                vals.append(kwargs[key])
        for json_key in self._INDEX_JSON_FIELDS:
            if json_key in kwargs:
                cols.append(json_key)
                vals.append(json.dumps(kwargs[json_key]))
        placeholders = ",".join("?" for _ in vals)
        col_str = ",".join(cols)
        await self._execute(
            f"INSERT INTO past_proposal_index ({col_str}) VALUES ({placeholders})",
            tuple(vals),
        )
        return await self.get_proposal_index(index_id)

    async def get_proposal_index(self, index_id: str) -> Optional[dict]:
        row = await self._fetchone(
            "SELECT * FROM past_proposal_index WHERE id=?", (index_id,)
        )
        return self._deserialize_index(row)

    async def get_proposal_index_by_folder(self, folder_name: str) -> Optional[dict]:
        row = await self._fetchone(
            "SELECT * FROM past_proposal_index WHERE folder_name=?", (folder_name,)
        )
        return self._deserialize_index(row)

    async def update_proposal_index(self, index_id: str, **kwargs) -> Optional[dict]:
        sets, vals = [], []
        for key, value in kwargs.items():
            if key in self._INDEX_JSON_FIELDS:
                value = json.dumps(value)
            sets.append(f"{key}=?")
            vals.append(value)
        sets.append("updated_at=datetime('now')")
        vals.append(index_id)
        await self._execute(
            f"UPDATE past_proposal_index SET {','.join(sets)} WHERE id=?",
            tuple(vals),
        )
        return await self.get_proposal_index(index_id)

    async def list_proposal_indexes(self) -> list[dict]:
        rows = await self._fetchall(
            "SELECT * FROM past_proposal_index ORDER BY indexed_at DESC"
        )
        for row in rows:
            self._deserialize_index(row)
        return rows

    async def search_proposal_index(
        self, query: str, sector: str = "", limit: int = 5
    ) -> list[dict]:
        match_expr = query
        if sector:
            match_expr = f"({query}) AND sector:{sector}"
        rows = await self._fetchall(
            "SELECT p.*, bm25(past_proposal_fts) AS rank "
            "FROM past_proposal_fts fts "
            "JOIN past_proposal_index p ON fts.rowid = p.rowid "
            "WHERE past_proposal_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT ?",
            (match_expr, limit),
        )
        for row in rows:
            self._deserialize_index(row)
        return rows

    async def delete_proposal_index(self, index_id: str) -> bool:
        existing = await self._fetchone(
            "SELECT id FROM past_proposal_index WHERE id=?", (index_id,)
        )
        if not existing:
            return False
        # Clean up vector embedding if vec is enabled
        if self._vec_enabled:
            row = await self._fetchone(
                "SELECT rowid FROM past_proposal_index WHERE id=?", (index_id,)
            )
            if row:
                await self._execute(
                    "DELETE FROM past_proposal_vec WHERE rowid=?", (row["rowid"],)
                )
        await self._execute(
            "DELETE FROM past_proposal_index WHERE id=?", (index_id,)
        )
        return True

    # ------------------------------------------------------------------
    # OAuth
    # ------------------------------------------------------------------

    async def save_oauth_client(
        self,
        client_id: str,
        client_secret: str,
        redirect_uris: str = "[]",
        client_name: str = "",
        grant_types: str = '["authorization_code"]',
        response_types: str = '["code"]',
        scope: str = "",
        token_endpoint_auth_method: str = "client_secret_post",
    ) -> None:
        await self._execute(
            "INSERT OR REPLACE INTO oauth_client "
            "(client_id, client_secret, redirect_uris, client_name, "
            "grant_types, response_types, scope, token_endpoint_auth_method) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (client_id, client_secret, redirect_uris, client_name,
             grant_types, response_types, scope, token_endpoint_auth_method),
        )

    async def get_oauth_client(self, client_id: str) -> Optional[dict]:
        return await self._fetchone(
            "SELECT * FROM oauth_client WHERE client_id=?", (client_id,)
        )

    async def save_oauth_auth_code(
        self,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
        expires_at: float,
        redirect_uri_provided_explicitly: bool = True,
        scope: str = "",
        resource: str = "",
    ) -> None:
        await self._execute(
            "INSERT INTO oauth_auth_code "
            "(code, client_id, redirect_uri, code_challenge, expires_at, "
            "redirect_uri_provided_explicitly, scope, resource) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (code, client_id, redirect_uri, code_challenge, expires_at,
             1 if redirect_uri_provided_explicitly else 0, scope, resource),
        )

    async def get_and_delete_oauth_auth_code(self, code: str) -> Optional[dict]:
        row = await self._fetchone(
            "SELECT * FROM oauth_auth_code WHERE code=?", (code,)
        )
        if row:
            await self._execute("DELETE FROM oauth_auth_code WHERE code=?", (code,))
        return row

    async def save_oauth_token(
        self,
        token: str,
        token_type: str,
        client_id: str,
        scope: str = "",
        resource: str = "",
        expires_at: Optional[int] = None,
    ) -> None:
        await self._execute(
            "INSERT INTO oauth_token (token, token_type, client_id, scope, resource, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (token, token_type, client_id, scope, resource, expires_at),
        )

    async def get_oauth_token(self, token: str) -> Optional[dict]:
        return await self._fetchone(
            "SELECT * FROM oauth_token WHERE token=?", (token,)
        )

    async def delete_oauth_token(self, token: str) -> None:
        await self._execute("DELETE FROM oauth_token WHERE token=?", (token,))

    # ------------------------------------------------------------------
    # Past Proposal Vectors (sqlite-vec)
    # ------------------------------------------------------------------

    async def upsert_proposal_vector(self, folder_name: str, embedding: list[float]) -> bool:
        """Store or replace the embedding vector for a past proposal.

        Links to past_proposal_index via rowid.
        """
        if not self._vec_enabled:
            return False
        row = await self._fetchone(
            "SELECT rowid FROM past_proposal_index WHERE folder_name=?", (folder_name,)
        )
        if not row:
            logger.warning("Cannot store vector — no index entry for folder: %s", folder_name)
            return False

        rowid = row["rowid"]
        blob = _serialize_vector(embedding)

        # Delete existing vector for this rowid, then insert new one
        await self._execute("DELETE FROM past_proposal_vec WHERE rowid=?", (rowid,))
        await self._execute(
            "INSERT INTO past_proposal_vec(rowid, embedding) VALUES (?, ?)",
            (rowid, blob),
        )
        logger.debug("Stored vector for %s (rowid=%d, dim=%d)", folder_name, rowid, len(embedding))
        return True

    async def search_proposal_vector(
        self, query_embedding: list[float], limit: int = 10
    ) -> list[dict]:
        """KNN search over proposal embeddings. Returns proposals ranked by distance.

        Results include all past_proposal_index fields plus a 'distance' column.
        """
        if not self._vec_enabled:
            return []
        blob = _serialize_vector(query_embedding)
        rows = await self._fetchall(
            "SELECT p.*, v.distance "
            "FROM past_proposal_vec v "
            "JOIN past_proposal_index p ON v.rowid = p.rowid "
            "WHERE v.embedding MATCH ? AND k = ?",
            (blob, limit),
        )
        for row in rows:
            self._deserialize_index(row)
        return rows
