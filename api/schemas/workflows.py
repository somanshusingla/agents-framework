from pydantic import BaseModel


class WorkflowInvokeRequest(BaseModel):
    thread_id: str
    input: str
