"""
PRODUCTION AUTH MIDDLEWARE
==========================

JWT verification middleware compatible with RG_Auth tokens.
Verifies HS256 JWT tokens using the shared secret key.

Supports:
- Bearer token in Authorization header
- Cookie-based auth (rg_access_token)
- Internal service-to-service auth via X-Internal-Key header
- WebSocket auth via query param (?token=...)

JWT payload structure from RG_Auth:
{
    "user_id": "uuid",
    "org_id": "uuid",
    "role": "user|org_admin|platform_dev|...",
    "scopes": ["..."],
    "auth_method": "jwt|api_key|internal",
    "token_version": int,
    "exp": timestamp,
    "iat": timestamp,
    "jti": "hex",
    "type": "access"
}
"""

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import Depends, HTTPException, Request, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# ── JWT verification (python-jose) ──
try:
    from jose import JWTError, jwt as jose_jwt
    HAS_JOSE = True
except ImportError:
    HAS_JOSE = False
    try:
        import jwt as pyjwt
        HAS_PYJWT = True
    except ImportError:
        HAS_PYJWT = False
        logger.warning("No JWT library found — install python-jose or PyJWT")

# ── Config ──
JWT_SECRET_KEY = os.getenv("AUTH_JWT_SECRET_KEY", "")
JWT_ALGORITHM = "HS256"
INTERNAL_SERVICE_KEY = os.getenv("AUTH_INTERNAL_SERVICE_KEY", "")
ACCESS_COOKIE_NAME = "rg_access_token"

# Dev mode: skip auth when no secret is configured (local testing only)
AUTH_ENABLED = bool(JWT_SECRET_KEY)

security_scheme = HTTPBearer(auto_error=False)


@dataclass
class AuthenticatedUser:
    """Verified identity from JWT or service key."""
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    email: Optional[str] = None
    role: str = "user"
    scopes: List[str] = field(default_factory=list)
    auth_method: str = "jwt"  # jwt | api_key | internal | dev
    miner_id: Optional[str] = None

    def has_scope(self, scope: str) -> bool:
        if "*" in self.scopes:
            return True
        return scope in self.scopes

    def is_admin(self) -> bool:
        return self.role in ("org_admin", "admin", "platform_dev", "system")

    def is_internal(self) -> bool:
        return self.auth_method == "internal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "org_id": self.org_id,
            "email": self.email,
            "role": self.role,
            "scopes": self.scopes,
            "auth_method": self.auth_method,
            "miner_id": self.miner_id,
        }


def _decode_jwt(token: str) -> Dict[str, Any]:
    """Decode and verify a JWT token."""
    if not JWT_SECRET_KEY:
        raise HTTPException(status_code=500, detail="JWT_SECRET_KEY not configured")

    try:
        if HAS_JOSE:
            payload = jose_jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        elif HAS_PYJWT:
            payload = pyjwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        else:
            raise HTTPException(status_code=500, detail="No JWT library available")

        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")

        return payload

    except Exception as e:
        if "expired" in str(e).lower():
            raise HTTPException(status_code=401, detail="Token expired")
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


def _extract_token(request: Request) -> Optional[str]:
    """Extract JWT from Authorization header or cookie."""
    # 1. Authorization: Bearer <token>
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    # 2. Cookie
    token = request.cookies.get(ACCESS_COOKIE_NAME)
    if token:
        return token

    return None


def _extract_ws_token(ws: WebSocket) -> Optional[str]:
    """Extract JWT from WebSocket query param or header."""
    # 1. Query param: ws://...?token=<jwt>
    token = ws.query_params.get("token")
    if token:
        return token

    # 2. Header (some WS clients support this)
    auth_header = ws.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    # 3. Cookie
    token = ws.cookies.get(ACCESS_COOKIE_NAME)
    if token:
        return token

    return None


async def get_current_user(request: Request) -> AuthenticatedUser:
    """
    FastAPI dependency: verify auth and return authenticated user.
    Use as: user: AuthenticatedUser = Depends(get_current_user)
    """
    # Check internal service key first
    internal_key = request.headers.get("X-Internal-Key", "")
    if internal_key and INTERNAL_SERVICE_KEY and internal_key == INTERNAL_SERVICE_KEY:
        return AuthenticatedUser(
            user_id="service",
            role="system",
            scopes=["*"],
            auth_method="internal",
        )

    # Dev mode: skip auth if no secret configured
    if not AUTH_ENABLED:
        return AuthenticatedUser(
            user_id="dev-user",
            org_id="dev-org",
            email="dev@localhost",
            role="platform_dev",
            scopes=["*"],
            auth_method="dev",
        )

    # Extract and verify JWT
    token = _extract_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = _decode_jwt(token)

    return AuthenticatedUser(
        user_id=payload.get("user_id"),
        org_id=payload.get("org_id"),
        email=payload.get("email"),
        role=payload.get("role", "user"),
        scopes=payload.get("scopes", []),
        auth_method=payload.get("auth_method", "jwt"),
    )


async def get_ws_user(ws: WebSocket) -> Optional[AuthenticatedUser]:
    """
    Verify WebSocket auth. Returns None if auth fails (caller should close WS).
    """
    # Dev mode
    if not AUTH_ENABLED:
        return AuthenticatedUser(
            user_id="dev-user",
            org_id="dev-org",
            email="dev@localhost",
            role="platform_dev",
            scopes=["*"],
            auth_method="dev",
        )

    token = _extract_ws_token(ws)
    if not token:
        return None

    try:
        payload = _decode_jwt(token)
        return AuthenticatedUser(
            user_id=payload.get("user_id"),
            org_id=payload.get("org_id"),
            email=payload.get("email"),
            role=payload.get("role", "user"),
            scopes=payload.get("scopes", []),
            auth_method=payload.get("auth_method", "jwt"),
        )
    except HTTPException:
        return None


# ── Rate Limiter ──

class RateLimiter:
    """
    In-memory sliding window rate limiter.
    Production should use Redis, but this works for single-instance.
    """

    def __init__(self):
        self._windows: Dict[str, List[float]] = defaultdict(list)
        self._last_cleanup = time.time()

    def check(self, key: str, max_requests: int, window_seconds: int) -> Tuple[bool, int]:
        """
        Check if request is allowed.
        Returns (allowed, remaining_requests).
        """
        now = time.time()
        cutoff = now - window_seconds

        # Periodic cleanup
        if now - self._last_cleanup > 60:
            self._cleanup(cutoff)

        # Remove expired entries
        self._windows[key] = [t for t in self._windows[key] if t > cutoff]

        if len(self._windows[key]) >= max_requests:
            return False, 0

        self._windows[key].append(now)
        remaining = max_requests - len(self._windows[key])
        return True, remaining

    def _cleanup(self, cutoff: float):
        """Remove expired keys."""
        dead_keys = [k for k, v in self._windows.items() if not v or v[-1] < cutoff]
        for k in dead_keys:
            del self._windows[k]
        self._last_cleanup = time.time()


# Global rate limiter
rate_limiter = RateLimiter()

# Rate limit configs per endpoint category
RATE_LIMITS = {
    "register": (10, 60),       # 10 per minute
    "heartbeat": (120, 60),     # 120 per minute (2/sec)
    "discover": (30, 60),       # 30 per minute
    "gradient_submit": (60, 60),# 60 per minute (1/sec)
    "task_assign": (30, 60),    # 30 per minute
    "genesis": (5, 300),        # 5 per 5 minutes
    "default": (60, 60),        # 60 per minute
}


def check_rate_limit(key: str, category: str = "default") -> int:
    """
    Check rate limit. Raises 429 if exceeded.
    Returns remaining requests.
    """
    max_req, window = RATE_LIMITS.get(category, RATE_LIMITS["default"])
    allowed, remaining = rate_limiter.check(key, max_req, window)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {window}s.",
            headers={"Retry-After": str(window)},
        )

    return remaining
