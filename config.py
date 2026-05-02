from pydantic import BaseModel


class Settings(BaseModel):
    db_url: str = "sqlite+aiosqlite:///./agent_framework.db"
    default_model: str = "gpt-4.1-mini"
