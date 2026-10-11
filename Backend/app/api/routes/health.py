import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from ...core import redis as redis_module

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health():
    try:
        pong = await redis_module.redis.ping()
        return {"status": "ok", "redis": bool(pong)}
    except RedisError:
        # Return 503 if Redis isn’t reachable. The exception text stays in the
        # server log; it can carry the Redis host and port.
        logger.exception("Health check: Redis ping failed")
        return JSONResponse({"status": "degraded", "redis": False, "detail": "Redis unreachable"}, status_code=503)
