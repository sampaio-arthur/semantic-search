from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from infrastructure.api.routers.api_router import compat_router, router
from infrastructure.config import get_settings
from infrastructure.db.session import init_db


def create_app() -> FastAPI:
    settings = get_settings()
    # vector_dim == 2**quantum_n_qubits is validated in Settings.__init__
    init_db(settings)

    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(compat_router)
    return app


app = create_app()
