from fastapi import APIRouter
from sqlalchemy import text

from agent_framework.persistence.db import engine

router = APIRouter()


@router.get('/health/live')
async def live():
    return {"status": "ok"}


@router.get('/health/ready')
async def ready():
    try:
        async with engine.connect() as conn:
            await conn.execute(text("select 1"))
        return {"status": "ready"}
    except Exception as exc:
        return {"status": "not_ready", "error": str(exc)}
