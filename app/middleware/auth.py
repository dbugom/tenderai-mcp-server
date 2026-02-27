"""Bearer token authentication middleware for HTTP transport.

Wraps the FastMCP Starlette ASGI app and validates the
Authorization: Bearer <token> header against MCP_API_KEY.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class BearerTokenMiddleware:
    """ASGI middleware that validates a static Bearer token.

    Passes through OPTIONS requests (CORS preflight) without auth.
    Returns 401 for missing/invalid tokens on all other requests.
    """

    def __init__(self, app, expected_token: str) -> None:
        self.app = app
        self.expected_token = expected_token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            # Pass through non-HTTP scopes (lifespan, websocket, etc.)
            await self.app(scope, receive, send)
            return

        # Allow CORS preflight without auth
        method = scope.get("method", "")
        if method == "OPTIONS":
            await self.app(scope, receive, send)
            return

        # Extract Authorization header
        headers = dict(scope.get("headers", []))
        auth_value = headers.get(b"authorization", b"").decode()

        if not auth_value.startswith("Bearer "):
            logger.warning("Missing or malformed Authorization header from %s", scope.get("client"))
            await self._send_401(send)
            return

        token = auth_value[7:]
        if token != self.expected_token:
            logger.warning("Invalid Bearer token from %s", scope.get("client"))
            await self._send_401(send)
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _send_401(send) -> None:
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b'Bearer error="invalid_token"'),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": b'{"error":"unauthorized","message":"Invalid or missing Bearer token"}',
        })
