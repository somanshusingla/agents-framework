from pydantic import BaseModel


class WorkflowInvokeRequest(BaseModel):
    thread_id: str
    input: str


class JobAcceptedResponse(BaseModel):
    job_id: str
    status: str = "queued"
