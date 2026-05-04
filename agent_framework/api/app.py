from fastapi import FastAPI
from agent_framework.api.routers.health import router as health_router
from agent_framework.api.routers.metrics import router as metrics_router
from agent_framework.api.routers.workflows import router as workflows_router
from agent_framework.persistence.init_db import init_db

app = FastAPI(title="Book Grounded LangGraph Agent")
app.include_router(workflows_router, prefix="/v1")
app.include_router(health_router)
app.include_router(metrics_router)


@app.on_event("startup")
async def startup() -> None:
    await init_db()
