"""Optional API key authentication independent of application users."""

from hmac import compare_digest


def authorize(scope: dict, api_keys: tuple[str, ...]) -> bool:
    """Verify a Bearer key or browser WebSocket credential.

    Parameters
    ----------
    scope : dict
        ASGI connection scope.
    api_keys : tuple[str, ...]
        Allowed keys, empty for unrestricted access.

    Returns
    -------
    bool
        Whether access is allowed.
    """
    if not api_keys:
        return True
    headers = dict(scope.get("headers", []))
    bearer = headers.get(b"authorization", b"")
    candidates = [bearer[7:]] if bearer.startswith(b"Bearer ") else []
    prefix = "openai-insecure-api-key."
    candidates.extend(
        p[len(prefix) :].encode()
        for p in scope.get("subprotocols", [])
        if p.startswith(prefix)
    )
    return any(
        compare_digest(candidate, key.encode())
        for candidate in candidates
        for key in api_keys
    )
