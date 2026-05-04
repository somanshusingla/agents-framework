from agent_framework.persistence.db import engine
from agent_framework.persistence.tables import Base


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
