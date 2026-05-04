from fastapi import APIRouter
from agent_framework.runtime.orchestrator import Orchestrator

router = APIRouter()
_orch = Orchestrator()


@router.get('/metrics')
async def metrics():
    total = len(_orch.jobs)
    completed = len([j for j in _orch.jobs.values() if j.get('status') == 'completed'])
    return {"jobs_total": total, "jobs_completed": completed}
