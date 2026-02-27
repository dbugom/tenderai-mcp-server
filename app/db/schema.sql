-- TenderAI SQLite Schema

CREATE TABLE IF NOT EXISTS rfp (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    client          TEXT NOT NULL,
    sector          TEXT DEFAULT 'telecom',
    country         TEXT DEFAULT 'OM',
    rfp_number      TEXT,
    issue_date      TEXT,
    deadline        TEXT,
    submission_method TEXT,
    status          TEXT DEFAULT 'new' CHECK(status IN ('new','analyzing','in_progress','submitted','awarded','lost','cancelled')),
    file_path       TEXT,
    parsed_sections TEXT DEFAULT '{}',
    requirements    TEXT DEFAULT '[]',
    evaluation_criteria TEXT DEFAULT '[]',
    notes           TEXT DEFAULT '',
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS proposal (
    id              TEXT PRIMARY KEY,
    rfp_id          TEXT NOT NULL REFERENCES rfp(id),
    proposal_type   TEXT NOT NULL CHECK(proposal_type IN ('technical','financial','combined')),
    status          TEXT DEFAULT 'draft' CHECK(status IN ('draft','review','final','submitted')),
    title           TEXT DEFAULT '',
    sections        TEXT DEFAULT '[]',
    output_path     TEXT,
    version         INTEGER DEFAULT 1,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS vendor (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    specialization  TEXT DEFAULT '',
    country         TEXT DEFAULT '',
    contact_name    TEXT DEFAULT '',
    contact_email   TEXT DEFAULT '',
    contact_phone   TEXT DEFAULT '',
    currency        TEXT DEFAULT 'USD',
    past_projects   TEXT DEFAULT '[]',
    notes           TEXT DEFAULT '',
    is_approved     INTEGER DEFAULT 0,
    rating          INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bom (
    id              TEXT PRIMARY KEY,
    proposal_id     TEXT NOT NULL REFERENCES proposal(id),
    category        TEXT NOT NULL,
    item_name       TEXT NOT NULL,
    description     TEXT DEFAULT '',
    vendor_id       TEXT REFERENCES vendor(id),
    manufacturer    TEXT DEFAULT '',
    part_number     TEXT DEFAULT '',
    quantity        REAL DEFAULT 1.0,
    unit            TEXT DEFAULT 'unit',
    unit_cost       REAL NOT NULL DEFAULT 0.0,
    margin_pct      REAL DEFAULT 15.0,
    total_cost      REAL GENERATED ALWAYS AS (quantity * unit_cost * (1 + margin_pct / 100.0)) STORED,
    warranty_months INTEGER DEFAULT 12,
    sort_order      INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS partner (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    country         TEXT DEFAULT '',
    specialization  TEXT DEFAULT '',
    contact_name    TEXT DEFAULT '',
    contact_email   TEXT DEFAULT '',
    contact_phone   TEXT DEFAULT '',
    nda_status      TEXT DEFAULT 'none' CHECK(nda_status IN ('none','sent','signed','expired')),
    nda_signed_date TEXT,
    nda_expiry_date TEXT,
    past_projects   TEXT DEFAULT '[]',
    notes           TEXT DEFAULT '',
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS partner_deliverable (
    id              TEXT PRIMARY KEY,
    partner_id      TEXT NOT NULL REFERENCES partner(id),
    proposal_id     TEXT NOT NULL REFERENCES proposal(id),
    title           TEXT NOT NULL,
    deliverable_type TEXT DEFAULT 'document' CHECK(deliverable_type IN ('technical_input','pricing','cv','reference_letter','certification','document','other')),
    due_date        TEXT,
    status          TEXT DEFAULT 'pending' CHECK(status IN ('pending','requested','in_progress','received','approved','overdue')),
    file_path       TEXT,
    notes           TEXT DEFAULT '',
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

-- Past Proposal Index — stores parsed & summarized metadata for fast search
CREATE TABLE IF NOT EXISTS past_proposal_index (
    id              TEXT PRIMARY KEY,
    folder_name     TEXT NOT NULL UNIQUE,
    tender_number   TEXT DEFAULT '',
    title           TEXT NOT NULL DEFAULT '',
    client          TEXT DEFAULT '',
    sector          TEXT DEFAULT '',
    country         TEXT DEFAULT '',
    technical_summary TEXT DEFAULT '',
    pricing_summary TEXT DEFAULT '',
    total_price     REAL DEFAULT 0.0,
    margin_info     TEXT DEFAULT '',
    technologies    TEXT DEFAULT '[]',
    keywords        TEXT DEFAULT '[]',
    full_summary    TEXT DEFAULT '',
    file_count      INTEGER DEFAULT 0,
    file_list       TEXT DEFAULT '[]',
    indexed_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

-- FTS5 virtual table for full-text search over past proposals
CREATE VIRTUAL TABLE IF NOT EXISTS past_proposal_fts USING fts5(
    title,
    client,
    sector,
    country,
    technical_summary,
    pricing_summary,
    technologies,
    keywords,
    full_summary,
    content='past_proposal_index',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 in sync with the main table
CREATE TRIGGER IF NOT EXISTS past_proposal_fts_insert
AFTER INSERT ON past_proposal_index BEGIN
    INSERT INTO past_proposal_fts(rowid, title, client, sector, country,
        technical_summary, pricing_summary, technologies, keywords, full_summary)
    VALUES (NEW.rowid, NEW.title, NEW.client, NEW.sector, NEW.country,
        NEW.technical_summary, NEW.pricing_summary, NEW.technologies,
        NEW.keywords, NEW.full_summary);
END;

CREATE TRIGGER IF NOT EXISTS past_proposal_fts_update
AFTER UPDATE ON past_proposal_index BEGIN
    INSERT INTO past_proposal_fts(past_proposal_fts, rowid, title, client, sector,
        country, technical_summary, pricing_summary, technologies, keywords, full_summary)
    VALUES ('delete', OLD.rowid, OLD.title, OLD.client, OLD.sector, OLD.country,
        OLD.technical_summary, OLD.pricing_summary, OLD.technologies,
        OLD.keywords, OLD.full_summary);
    INSERT INTO past_proposal_fts(rowid, title, client, sector, country,
        technical_summary, pricing_summary, technologies, keywords, full_summary)
    VALUES (NEW.rowid, NEW.title, NEW.client, NEW.sector, NEW.country,
        NEW.technical_summary, NEW.pricing_summary, NEW.technologies,
        NEW.keywords, NEW.full_summary);
END;

CREATE TRIGGER IF NOT EXISTS past_proposal_fts_delete
AFTER DELETE ON past_proposal_index BEGIN
    INSERT INTO past_proposal_fts(past_proposal_fts, rowid, title, client, sector,
        country, technical_summary, pricing_summary, technologies, keywords, full_summary)
    VALUES ('delete', OLD.rowid, OLD.title, OLD.client, OLD.sector, OLD.country,
        OLD.technical_summary, OLD.pricing_summary, OLD.technologies,
        OLD.keywords, OLD.full_summary);
END;

-- OAuth 2.0 tables (Dynamic Client Registration, Authorization Codes, Tokens)

CREATE TABLE IF NOT EXISTS oauth_client (
    client_id       TEXT PRIMARY KEY,
    client_secret   TEXT NOT NULL,
    redirect_uris   TEXT NOT NULL DEFAULT '[]',
    client_name     TEXT DEFAULT '',
    grant_types     TEXT DEFAULT '["authorization_code"]',
    response_types  TEXT DEFAULT '["code"]',
    scope           TEXT DEFAULT '',
    token_endpoint_auth_method TEXT DEFAULT 'client_secret_post',
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS oauth_auth_code (
    code            TEXT PRIMARY KEY,
    client_id       TEXT NOT NULL,
    redirect_uri    TEXT NOT NULL,
    redirect_uri_provided_explicitly INTEGER DEFAULT 1,
    scope           TEXT DEFAULT '',
    code_challenge  TEXT NOT NULL,
    resource        TEXT DEFAULT '',
    expires_at      REAL NOT NULL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS oauth_token (
    token           TEXT PRIMARY KEY,
    token_type      TEXT NOT NULL CHECK(token_type IN ('access','refresh')),
    client_id       TEXT NOT NULL,
    scope           TEXT DEFAULT '',
    resource        TEXT DEFAULT '',
    expires_at      INTEGER,
    created_at      TEXT DEFAULT (datetime('now'))
);
