"""ASGI security boundary for the desktop loopback service."""

from __future__ import annotations

import ipaddress
import json
import secrets
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import parse_qs

from .config import normalize_origin


STARTUP_TOKEN_HEADER = "X-XTalk-App-Token"
STARTUP_TOKEN_QUERY = "app_token"
CORE_ACCESS_TOKEN_QUERY = "access_token"

ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGISend = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[dict[str, Any], ASGIReceive, ASGISend], Awaitable[None]]


@dataclass(frozen=True)
class SecurityDenial:
    """Describe why an ASGI request must not reach the application.

    Parameters
    ----------
    status_code : int
        HTTP status used for HTTP requests.
    message : str
        Non-sensitive client-facing error message.
    """

    status_code: int
    message: str


def is_loopback_host(host: str | None) -> bool:
    """Return whether an ASGI peer host is local loopback.

    Parameters
    ----------
    host : str | None
        Peer address reported in the ASGI client tuple.

    Returns
    -------
    bool
        ``True`` only for localhost or an IP loopback address.
    """

    if host is None:
        return False
    normalized = host.strip().strip("[]")
    if normalized.lower() == "localhost":
        return True
    normalized = normalized.split("%", 1)[0]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _headers(scope: Mapping[str, Any]) -> dict[str, str]:
    """Decode ASGI headers into a lower-case mapping."""

    decoded: dict[str, str] = {}
    for raw_name, raw_value in scope.get("headers", []):
        decoded[raw_name.decode("latin-1").lower()] = raw_value.decode("latin-1")
    return decoded


def _query(scope: Mapping[str, Any]) -> dict[str, list[str]]:
    """Decode an ASGI query string without logging its secret values."""

    raw_query = scope.get("query_string", b"")
    if not isinstance(raw_query, bytes):
        return {}
    return parse_qs(
        raw_query.decode("utf-8", errors="replace"),
        keep_blank_values=True,
    )


def _bearer_token(authorization: str | None) -> str | None:
    """Extract a bearer token from one Authorization header."""

    if not authorization:
        return None
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


class SecurityPolicy:
    """Authorize requests against loopback, Origin, and launch-token rules.

    Parameters
    ----------
    token : str
        Per-launch startup token.
    origins : tuple[str, ...]
        Exact normalized WebView origins permitted by the parent process.
    """

    def __init__(self, *, token: str, origins: tuple[str, ...]) -> None:
        self._token_bytes = token.encode("utf-8")
        self._origins = frozenset(origins)

    def evaluate(self, scope: Mapping[str, Any]) -> SecurityDenial | None:
        """Return a denial for an unauthorized ASGI scope.

        Parameters
        ----------
        scope : Mapping[str, Any]
            HTTP or WebSocket ASGI scope.

        Returns
        -------
        SecurityDenial | None
            Denial reason, or ``None`` when the request may continue.
        """

        scope_type = scope.get("type")
        if scope_type not in {"http", "websocket"}:
            return None

        client = scope.get("client")
        peer_host = (
            client[0]
            if isinstance(client, (list, tuple)) and client
            else None
        )
        if not is_loopback_host(peer_host):
            return SecurityDenial(403, "Loopback clients only")

        request_headers = _headers(scope)
        origin = request_headers.get("origin")
        if origin is not None:
            try:
                normalized_origin = normalize_origin(origin)
            except ValueError:
                return SecurityDenial(403, "Origin is not allowed")
            if normalized_origin not in self._origins:
                return SecurityDenial(403, "Origin is not allowed")

        path = str(scope.get("path", ""))
        query = _query(scope)
        if scope_type == "http":
            method = str(scope.get("method", "GET")).upper()
            if method == "OPTIONS":
                if origin is None:
                    return SecurityDenial(
                        403,
                        "Origin is required for preflight",
                    )
                return None
            if self._has_path_prefix(path, "/app"):
                if not self._has_startup_token(
                    request_headers,
                    query,
                    allow_authorization=True,
                ):
                    return SecurityDenial(401, "Startup token is required")
            elif self._has_path_prefix(path, "/api"):
                # Authorization belongs to XTalk's JWT authentication on these
                # routes, so the launch token uses its dedicated header/query.
                if not self._has_startup_token(
                    request_headers,
                    query,
                    allow_authorization=False,
                ):
                    return SecurityDenial(401, "Startup token is required")
            elif not self._has_startup_token(
                request_headers,
                query,
                allow_authorization=True,
            ):
                return SecurityDenial(401, "Startup token is required")
            return None

        if path == "/ws":
            if self._has_startup_token(
                request_headers,
                query,
                allow_authorization=False,
            ):
                return None

            # The current public browser SDK adds only `access_token` to its
            # WebSocket URL and cannot attach the app-specific launch header.
            # Presence is enough at this outer boundary; XTalk's public route
            # remains responsible for cryptographically validating the JWT.
            access_tokens = query.get(CORE_ACCESS_TOKEN_QUERY, [])
            if any(token for token in access_tokens):
                return None
            return SecurityDenial(401, "Startup or access token is required")

        if not self._has_startup_token(
            request_headers,
            query,
            allow_authorization=False,
        ):
            return SecurityDenial(401, "Startup token is required")
        return None

    @staticmethod
    def _has_path_prefix(path: str, prefix: str) -> bool:
        """Return whether a URL path equals a protected prefix or is below it."""

        return path == prefix or path.startswith(f"{prefix}/")

    def _has_startup_token(
        self,
        headers: Mapping[str, str],
        query: Mapping[str, list[str]],
        *,
        allow_authorization: bool,
    ) -> bool:
        """Return whether request metadata contains the exact launch token."""

        candidates = [
            headers.get(STARTUP_TOKEN_HEADER.lower()),
            *query.get(STARTUP_TOKEN_QUERY, []),
        ]
        if allow_authorization:
            candidates.append(_bearer_token(headers.get("authorization")))
        return any(
            candidate is not None
            and secrets.compare_digest(
                candidate.encode("utf-8"),
                self._token_bytes,
            )
            for candidate in candidates
        )


class SidecarSecurityMiddleware:
    """Reject non-loopback, cross-origin, and unauthenticated ASGI requests.

    Parameters
    ----------
    app : ASGIApp
        Wrapped FastAPI application.
    token : str
        Per-launch startup token.
    origins : tuple[str, ...]
        Exact allowed WebView origins.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        token: str,
        origins: tuple[str, ...],
    ) -> None:
        self._app = app
        self._policy = SecurityPolicy(token=token, origins=origins)

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        """Apply the security policy and dispatch an authorized request."""

        denial = self._policy.evaluate(scope)
        if denial is None:
            await self._app(scope, receive, send)
            return

        if scope.get("type") == "websocket":
            await send(
                {
                    "type": "websocket.close",
                    "code": 1008,
                    "reason": denial.message,
                }
            )
            return

        body = json.dumps(
            {"detail": denial.message},
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": denial.status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
