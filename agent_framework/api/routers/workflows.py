import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from agent_framework.api.runtime import orchestrator
from agent_framework.api.schemas.workflows import JobAcceptedResponse, WorkflowInvokeRequest

router = APIRouter()


@router.post("/workflows/default/invoke")
async def invoke_workflow(request: WorkflowInvokeRequest):
    return await orchestrator.invoke(request.thread_id, request.input)


@router.post("/workflows/default/jobs", response_model=JobAcceptedResponse)
async def submit_job(request: WorkflowInvokeRequest):
    job_id = await orchestrator.submit_job(request.thread_id, request.input)
    return JobAcceptedResponse(job_id=job_id)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    return orchestrator.jobs.get(job_id, {"status": "not_found"})


@router.post("/workflows/default/stream")
async def stream_workflow(request: WorkflowInvokeRequest):
    async def event_gen():
        async for event in orchestrator.stream(request.thread_id, request.input):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
