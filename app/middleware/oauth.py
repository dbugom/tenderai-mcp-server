"""OAuth 2.0 Authorization Server Provider for TenderAI.

Implements the MCP SDK's OAuthAuthorizationServerProvider interface so that
claude.ai (and other OAuth 2.0 clients) can connect via Dynamic Client
Registration + Authorization Code + PKCE.

Auto-approves all authorization requests (single-user private server).
"""

from __future__ import annotations

import json
import logging
import secrets
import time

from pydantic import AnyUrl

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from app.db.database import Database

logger = logging.getLogger(__name__)

# Token lifetimes
ACCESS_TOKEN_TTL = 3600  # 1 hour
REFRESH_TOKEN_TTL = 30 * 24 * 3600  # 30 days
AUTH_CODE_TTL = 300  # 5 minutes


class TenderAIOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    """OAuth provider backed by SQLite.

    All authorization requests are auto-approved (no login UI) since TenderAI
    is a private single-user server.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    # ---- Client Registration (RFC 7591) ----

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        row = await self.db.get_oauth_client(client_id)
        if not row:
            return None
        return OAuthClientInformationFull(
            client_id=row["client_id"],
            client_secret=row["client_secret"],
            redirect_uris=[AnyUrl(u) for u in json.loads(row["redirect_uris"])],
            client_name=row["client_name"] or None,
            grant_types=json.loads(row["grant_types"]),
            response_types=json.loads(row["response_types"]),
            scope=row["scope"] or None,
            token_endpoint_auth_method=row["token_endpoint_auth_method"] or None,
        )

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        await self.db.save_oauth_client(
            client_id=client_info.client_id,
            client_secret=client_info.client_secret or "",
            redirect_uris=json.dumps([str(u) for u in client_info.redirect_uris]),
            client_name=client_info.client_name or "",
            grant_types=json.dumps(client_info.grant_types),
            response_types=json.dumps(client_info.response_types),
            scope=client_info.scope or "",
            token_endpoint_auth_method=client_info.token_endpoint_auth_method or "client_secret_post",
        )
        logger.info("Registered OAuth client: %s (%s)", client_info.client_id, client_info.client_name)

    # ---- Authorization (auto-approve) ----

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        code = secrets.token_hex(32)  # 256 bits of entropy
        expires_at = time.time() + AUTH_CODE_TTL

        scopes = params.scopes or []

        await self.db.save_oauth_auth_code(
            code=code,
            client_id=client.client_id,
            redirect_uri=str(params.redirect_uri),
            code_challenge=params.code_challenge,
            expires_at=expires_at,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            scope=" ".join(scopes),
            resource=params.resource or "",
        )

        logger.info("Auto-approved authorization for client %s", client.client_id)

        return construct_redirect_uri(
            str(params.redirect_uri),
            code=code,
            state=params.state,
        )

    # ---- Authorization Code ----

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        row = await self.db.get_and_delete_oauth_auth_code(authorization_code)
        if not row:
            return None
        if row["client_id"] != client.client_id:
            logger.warning("Auth code client mismatch: expected %s, got %s", client.client_id, row["client_id"])
            return None
        return AuthorizationCode(
            code=row["code"],
            client_id=row["client_id"],
            redirect_uri=AnyUrl(row["redirect_uri"]),
            redirect_uri_provided_explicitly=bool(row["redirect_uri_provided_explicitly"]),
            scopes=row["scope"].split() if row["scope"] else [],
            code_challenge=row["code_challenge"],
            expires_at=row["expires_at"],
            resource=row["resource"] or None,
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        access_token = secrets.token_hex(32)
        refresh_token = secrets.token_hex(32)
        scope_str = " ".join(authorization_code.scopes) if authorization_code.scopes else ""
        resource = authorization_code.resource or ""

        now = int(time.time())
        access_expires = now + ACCESS_TOKEN_TTL
        refresh_expires = now + REFRESH_TOKEN_TTL

        await self.db.save_oauth_token(
            token=access_token,
            token_type="access",
            client_id=client.client_id,
            scope=scope_str,
            resource=resource,
            expires_at=access_expires,
        )
        await self.db.save_oauth_token(
            token=refresh_token,
            token_type="refresh",
            client_id=client.client_id,
            scope=scope_str,
            resource=resource,
            expires_at=refresh_expires,
        )

        logger.info("Issued tokens for client %s", client.client_id)

        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL,
            scope=scope_str or None,
            refresh_token=refresh_token,
        )

    # ---- Refresh Token ----

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        row = await self.db.get_oauth_token(refresh_token)
        if not row or row["token_type"] != "refresh":
            return None
        if row["client_id"] != client.client_id:
            return None
        if row["expires_at"] and int(time.time()) > row["expires_at"]:
            await self.db.delete_oauth_token(refresh_token)
            return None
        return RefreshToken(
            token=row["token"],
            client_id=row["client_id"],
            scopes=row["scope"].split() if row["scope"] else [],
            expires_at=row["expires_at"],
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        # Delete old refresh token
        await self.db.delete_oauth_token(refresh_token.token)

        # Use requested scopes or fall back to original
        new_scopes = scopes if scopes else refresh_token.scopes
        scope_str = " ".join(new_scopes) if new_scopes else ""
        resource = ""

        # Try to get resource from the old refresh token row (already deleted,
        # but we have it in the RefreshToken model — no resource field there,
        # so we leave it empty; the access token will inherit from scope context)

        new_access = secrets.token_hex(32)
        new_refresh = secrets.token_hex(32)

        now = int(time.time())

        await self.db.save_oauth_token(
            token=new_access,
            token_type="access",
            client_id=client.client_id,
            scope=scope_str,
            resource=resource,
            expires_at=now + ACCESS_TOKEN_TTL,
        )
        await self.db.save_oauth_token(
            token=new_refresh,
            token_type="refresh",
            client_id=client.client_id,
            scope=scope_str,
            resource=resource,
            expires_at=now + REFRESH_TOKEN_TTL,
        )

        logger.info("Refreshed tokens for client %s", client.client_id)

        return OAuthToken(
            access_token=new_access,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL,
            scope=scope_str or None,
            refresh_token=new_refresh,
        )

    # ---- Access Token ----

    async def load_access_token(self, token: str) -> AccessToken | None:
        row = await self.db.get_oauth_token(token)
        if not row or row["token_type"] != "access":
            return None
        if row["expires_at"] and int(time.time()) > row["expires_at"]:
            await self.db.delete_oauth_token(token)
            return None
        return AccessToken(
            token=row["token"],
            client_id=row["client_id"],
            scopes=row["scope"].split() if row["scope"] else [],
            expires_at=row["expires_at"],
            resource=row["resource"] or None,
        )

    # ---- Revocation ----

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        await self.db.delete_oauth_token(token.token)
        logger.info("Revoked %s token for client %s", type(token).__name__, token.client_id)
