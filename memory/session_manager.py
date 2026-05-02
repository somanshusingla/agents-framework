from __future__ import annotations

import json
from sqlalchemy import select
from persistence.db import SessionLocal
from persistence.tables import SessionTable, EventTable
from runtime.state import ExecutionEvent


class SessionManager:
    async def load_events(self, thread_id: str) -> list[ExecutionEvent]:
        async with SessionLocal() as db:
            q = await db.execute(select(EventTable).where(EventTable.thread_id == thread_id).order_by(EventTable.seq_no.asc()))
            rows = q.scalars().all()
        return [ExecutionEvent(id=r.id, thread_id=r.thread_id, run_id=r.run_id, seq_no=r.seq_no, ts=r.ts, event_type=r.event_type, payload=json.loads(r.payload_json)) for r in rows]

    async def append_event(self, event: ExecutionEvent, workflow_name: str = "default", agent_name: str = "default") -> None:
        async with SessionLocal() as db:
            sess = await db.get(SessionTable, event.thread_id)
            if not sess:
                sess = SessionTable(thread_id=event.thread_id, workflow_name=workflow_name, agent_name=agent_name)
                db.add(sess)
            db.add(EventTable(id=event.id, thread_id=event.thread_id, run_id=event.run_id, seq_no=event.seq_no, event_type=event.event_type, payload_json=json.dumps(event.payload), ts=event.ts))
            await db.commit()
