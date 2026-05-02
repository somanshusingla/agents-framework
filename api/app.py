from fastapi import FastAPI
from api.routers.workflows import router as workflows_router
from api.routers.health import router as health_router

app = FastAPI(title="Book Grounded LangGraph Agent")
app.include_router(workflows_router, prefix="/v1")
app.include_router(health_router)
