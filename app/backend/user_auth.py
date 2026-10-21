import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


class JWTAuth:
    """Minimal HS256 JWT helper used by the built-in auth routes."""

    def __init__(self, *, secret: str, ttl_seconds: int = 7 * 24 * 60 * 60):
        self._secret = secret.encode("utf-8")
        self._ttl_seconds = ttl_seconds

    def issue_token(self, *, sub: str) -> str:
        now = int(time.time())
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {"sub": sub, "iat": now, "exp": now + self._ttl_seconds}
        encoded_header = _b64url_encode(
            json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        encoded_payload = _b64url_encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
        signature = hmac.new(
            self._secret, signing_input, digestmod=hashlib.sha256
        ).digest()
        return f"{encoded_header}.{encoded_payload}.{_b64url_encode(signature)}"

    def verify_token(self, token: str) -> str:
        try:
            encoded_header, encoded_payload, encoded_signature = token.split(".")
        except ValueError as exc:
            raise ValueError("Malformed token") from exc

        signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
        expected = hmac.new(
            self._secret, signing_input, digestmod=hashlib.sha256
        ).digest()
        received = _b64url_decode(encoded_signature)
        if not hmac.compare_digest(expected, received):
            raise ValueError("Invalid token signature")

        try:
            payload = json.loads(_b64url_decode(encoded_payload))
        except Exception as exc:
            raise ValueError("Invalid token payload") from exc

        exp = int(payload.get("exp", 0))
        sub = str(payload.get("sub", "")).strip()
        if not sub:
            raise ValueError("Missing token subject")
        if exp and exp < int(time.time()):
            raise ValueError("Token expired")
        return sub


def extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def resolve_auth_config(service_config: dict[str, Any]) -> tuple[str, int]:
    auth_secret = str(
        service_config.get("auth_secret")
        or os.environ.get("XTALK_AUTH_SECRET")
        or secrets.token_hex(32)
    )
    auth_ttl_seconds = int(service_config.get("auth_token_ttl_seconds", 604800))
    return auth_secret, auth_ttl_seconds
