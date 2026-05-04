from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

db_url = os.getenv("AGENT_FRAMEWORK_DB_URL", "sqlite+aiosqlite:///./agent_framework.db")
engine = create_async_engine(db_url, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
