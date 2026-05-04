from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agent_framework.config import PersistenceSpec
from agent_framework.persistence.db import SessionLocal, db_url as default_db_url, engine as default_engine
from agent_framework.persistence.tables import Base, EventTable, SessionTable
from agent_framework.runtime.state import ExecutionEvent


class SessionManagerProtocol(Protocol):
    async def load_events(self, thread_id: str) -> list[ExecutionEvent]: ...

    async def append_event(
        self,
        event: ExecutionEvent,
        workflow_name: str = "default",
        agent_name: str = "default",
    ) -> None: ...


class SessionManager:
    """SQLite-backed session/event persistence."""

    def __init__(self, db_url: str | None = None, *, auto_create: bool = True) -> None:
        self.db_url = db_url or default_db_url
        self._auto_create = auto_create
        self._initialized = False
        if db_url:
            self._engine = create_async_engine(db_url, future=True)
            self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        else:
            self._engine = default_engine
            self._session_factory = SessionLocal

    async def load_events(self, thread_id: str) -> list[ExecutionEvent]:
        await self._ensure_initialized()
        async with self._session_factory() as db:
            q = await db.execute(
                select(EventTable)
                .where(EventTable.thread_id == thread_id)
                .order_by(EventTable.seq_no.asc())
            )
            rows = q.scalars().all()
        return [
            ExecutionEvent(
                id=r.id,
                thread_id=r.thread_id,
                run_id=r.run_id,
                seq_no=r.seq_no,
                ts=r.ts,
                event_type=r.event_type,
                payload=json.loads(r.payload_json),
            )
            for r in rows
        ]

    async def append_event(
        self,
        event: ExecutionEvent,
        workflow_name: str = "default",
        agent_name: str = "default",
    ) -> None:
        await self._ensure_initialized()
        async with self._session_factory() as db:
            sess = await db.get(SessionTable, event.thread_id)
            if not sess:
                sess = SessionTable(
                    thread_id=event.thread_id,
                    workflow_name=workflow_name,
                    agent_name=agent_name,
                )
                db.add(sess)
            db.add(
                EventTable(
                    id=event.id,
                    thread_id=event.thread_id,
                    run_id=event.run_id,
                    seq_no=event.seq_no,
                    event_type=event.event_type,
                    payload_json=json.dumps(event.payload),
                    ts=event.ts,
                )
            )
            await db.commit()

    async def _ensure_initialized(self) -> None:
        if self._initialized or not self._auto_create:
            return
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._initialized = True


class InMemorySessionManager:
    def __init__(self) -> None:
        self.events: list[ExecutionEvent] = []

    async def load_events(self, thread_id: str) -> list[ExecutionEvent]:
        return [event for event in self.events if event.thread_id == thread_id]

    async def append_event(
        self,
        event: ExecutionEvent,
        workflow_name: str = "default",
        agent_name: str = "default",
    ) -> None:
        self.events.append(event)


class NoOpSessionManager:
    async def load_events(self, thread_id: str) -> list[ExecutionEvent]:
        return []

    async def append_event(
        self,
        event: ExecutionEvent,
        workflow_name: str = "default",
        agent_name: str = "default",
    ) -> None:
        return None


SessionManagerFactoryFunc = Callable[[PersistenceSpec], SessionManagerProtocol]


class SessionManagerFactory:
    def __init__(self) -> None:
        self._backends: dict[str, SessionManagerFactoryFunc] = {}
        self.register("sqlite", lambda spec: SessionManager(spec.db_url))
        self.register("persistent", lambda spec: SessionManager(spec.db_url))
        self.register("memory", lambda spec: InMemorySessionManager())
        self.register("in_memory", lambda spec: InMemorySessionManager())
        self.register("noop", lambda spec: NoOpSessionManager())
        self.register("no_op", lambda spec: NoOpSessionManager())

    def register(
        self,
        backend: str,
        factory: SessionManagerFactoryFunc,
        *,
        replace: bool = False,
    ) -> None:
        key = _normalize_backend(backend)
        if key in self._backends and not replace:
            raise ValueError(f"Session backend already registered: {backend}")
        self._backends[key] = factory

    def backend(self, backend: str, *, replace: bool = False):
        def decorator(factory: SessionManagerFactoryFunc) -> SessionManagerFactoryFunc:
            self.register(backend, factory, replace=replace)
            return factory

        return decorator

    def create(self, spec: PersistenceSpec) -> SessionManagerProtocol:
        key = _normalize_backend(spec.backend)
        try:
            factory = self._backends[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._backends))
            raise ValueError(
                f"Unknown session backend '{spec.backend}'. Registered backends: {available}"
            ) from exc
        return factory(spec)


def _normalize_backend(backend: str) -> str:
    return backend.strip().lower().replace("-", "_")


default_session_manager_factory = SessionManagerFactory()


def create_session_manager(
    spec: PersistenceSpec,
    *,
    factory: SessionManagerFactory | None = None,
) -> SessionManagerProtocol:
    return (factory or default_session_manager_factory).create(spec)


def register_session_backend(
    backend: str,
    factory: SessionManagerFactoryFunc,
    *,
    replace: bool = False,
) -> None:
    default_session_manager_factory.register(backend, factory, replace=replace)


def session_backend(backend: str, *, replace: bool = False):
    return default_session_manager_factory.backend(backend, replace=replace)
