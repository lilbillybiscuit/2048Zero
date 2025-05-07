"""
Authentication middleware for the 2048-Zero Trainer Server
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for token-based authentication"""
    
    def __init__(self, app: FastAPI, token: str):
        """Initialize the middleware
        
        Args:
            app: FastAPI application
            token: Authentication token
        """
        super().__init__(app)
        self.token = token
    
    async def dispatch(self, request: Request, call_next):
        """Process each request
        
        Args:
            request: HTTP request
            call_next: Next middleware in chain
        """
        # Skip authentication for docs
        if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi"):
            return await call_next(request)
        
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content="Missing Authorization header",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Validate token
        try:
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer":
                raise ValueError("Invalid authentication scheme")
            if token != self.token:
                raise ValueError("Invalid token")
        except (ValueError, IndexError):
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Token is valid, proceed
        return await call_next(request)


# FastAPI security scheme
security = HTTPBearer()

def get_token_from_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to extract token from authorization header
    
    Args:
        credentials: Authorization credentials
    
    Returns:
        The token string
    """
    return credentials.credentials