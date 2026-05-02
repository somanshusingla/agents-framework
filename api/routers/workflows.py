from fastapi import APIRouter
from api.schemas.workflows import WorkflowInvokeRequest
from runtime.orchestrator import Orchestrator

router = APIRouter()
_orchestrator = Orchestrator()


@router.post("/workflows/default/invoke")
async def invoke_workflow(request: WorkflowInvokeRequest):
    return await _orchestrator.invoke(request.thread_id, request.input)
