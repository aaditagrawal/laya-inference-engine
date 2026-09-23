"""HTTP serving with one engine instance and CPU request validation."""

from contextlib import asynccontextmanager
import hmac
import logging
import os
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("laya_blackwell.server")


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    state: str | dict[str, Any] | list[Any]
    questions: dict[str, dict[str, Any]]


def _load_engine(**kwargs):
    # Keep importing this module and testing HTTP behavior independent of CUDA.
    from .engine import BlackwellEngine

    return BlackwellEngine(**kwargs)


def _warmup_requests():
    state = "The customer received a damaged order and is asking for a replacement."
    ready = {"type": "noul", "instructions": "Does the customer need help?"}
    route = {
        "type": "choice",
        "instructions": "Choose the team that should handle this request.",
        "criteria": ["support", "sales", "billing"],
    }
    urgency = {
        "type": "score",
        "instructions": "Rate how urgently the request needs a response.",
        "criteria": ["low", "normal", "high"],
    }
    return [
        (state, {"needs_help": ready}),
        (state, {"route": route}),
        (state, {"needs_help": ready, "route": route, "urgency": urgency}),
    ]


def create_app(
    engine=None,
    *,
    model: str | None = None,
    revision: str | None = None,
    backend: str = "fused",
    device: str = "cuda:0",
    api_key: str | None = None,
    warmup: bool = True,
) -> FastAPI:
    """Create an app and load its engine during startup if none is supplied.

    Injected engines remain caller-owned: this app does not warm or close them.
    The health route is public. When configured, the bearer key protects
    inference. A supplied key takes precedence over LAYA_API_KEY.
    """
    key = api_key if api_key is not None else os.environ.get("LAYA_API_KEY")
    owned = engine is None
    options = {"backend": backend, "device": device}
    if model is not None:
        options["model"] = model
    if revision is not None:
        options["revision"] = revision

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active = engine
        try:
            if owned:
                active = await run_in_threadpool(_load_engine, **options)
            app.state.engine = active
            if owned and warmup:
                for state, questions in _warmup_requests():
                    metrics = await run_in_threadpool(active.warmup, state, questions)
                    logger.info("Warmup completed: %s", metrics)
            logger.info(
                "Laya ready on %s with backend=%s.",
                getattr(active, "device", device),
                getattr(active, "backend", backend),
            )
            if getattr(active, "backend", backend) in {"fused", "fp8"}:
                logger.info(
                    "Warmup covers representative requests only; unseen batch, sequence, "
                    "or option shapes can incur cold kernel compilation and CUDA graph "
                    "capture. Responses report graph_miss and graph_build_ms."
                )
            yield
        finally:
            app.state.engine = None
            if owned and active is not None:
                await run_in_threadpool(active.close)

    app = FastAPI(title="Laya Blackwell", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    bearer = HTTPBearer(auto_error=False)

    async def authorize(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ):
        if key and (
            credentials is None
            or not hmac.compare_digest(credentials.credentials.encode(), key.encode())
        ):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/health")
    async def health():
        active = app.state.engine
        if active is None or getattr(active, "closed", False):
            raise HTTPException(status_code=503, detail="Engine is unavailable")
        return {"status": "ok", "backend": getattr(active, "backend", backend)}

    @app.post("/v1/systemone", dependencies=[Depends(authorize)])
    async def system_one(request: SystemOneRequest):
        active = app.state.engine
        if active is None or getattr(active, "closed", False):
            raise HTTPException(status_code=503, detail="Engine is unavailable")
        try:
            return await run_in_threadpool(active.predict, request.state, request.questions)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
