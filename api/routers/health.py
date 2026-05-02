from fastapi import APIRouter

router = APIRouter()


@router.get('/health/live')
async def live():
    return {"status": "ok"}
