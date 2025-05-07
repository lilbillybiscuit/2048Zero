from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class TokenAuthMiddleware(BaseHTTPMiddleware):
    
    def __init__(self, app: FastAPI, token: str):
        super().__init__(app)
        self.token = token
    
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content="Missing Authorization header",
                headers={"WWW-Authenticate": "Bearer"}
            )

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

        return await call_next(request)


security = HTTPBearer()

def get_token_from_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return credentials.credentials