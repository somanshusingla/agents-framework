from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from config import Settings

settings = Settings()
engine = create_async_engine(settings.db_url, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
