"""FastAPI application entry point."""

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routers import health, analyse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Speech Slurring Detection API",
        description="Clinical-grade dysarthria detection for stroke early warning",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router)
    app.include_router(analyse.router)

    # Serve static files (React build)
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{full_path:path}")
        async def serve_react_app(full_path: str):
            """Serve React app for all non-API routes."""
            # Skip API routes
            if full_path.startswith(("v1/", "healthz", "readyz", "docs", "redoc", "openapi.json")):
                return None

            # Serve index.html for all other routes (React routing)
            index_file = static_dir / "index.html"
            if index_file.exists():
                return FileResponse(index_file)

            return {"message": "Frontend not built. Run build_frontend.sh"}

    @app.on_event("startup")
    async def startup_event():
        """Startup tasks."""
        logger.info("=" * 80)
        logger.info("🚀 Speech Slurring Detection API starting...")
        logger.info("=" * 80)

    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown tasks."""
        logger.info("Shutting down API...")

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
