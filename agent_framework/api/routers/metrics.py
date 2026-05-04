from fastapi import APIRouter
from agent_framework.api.runtime import orchestrator

router = APIRouter()


@router.get('/metrics')
async def metrics():
    total = len(orchestrator.jobs)
    completed = len([j for j in orchestrator.jobs.values() if j.get('status') == 'completed'])
    return {"jobs_total": total, "jobs_completed": completed}
