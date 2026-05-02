from pydantic import BaseModel, Field


class WorkflowInvokeRequest(BaseModel):
    thread_id: str
    input: str


class JobAcceptedResponse(BaseModel):
    job_id: str
    status: str = "queued"


class ApprovalDecision(BaseModel):
    approved: bool
    reason: str | None = None


class ApprovalRequest(BaseModel):
    call_id: str
    decision: ApprovalDecision


class WorkflowInvokeResponse(BaseModel):
    run_id: str
    thread_id: str
    status: str
    output: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
