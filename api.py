from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import APIConfig
from routes import get_search_router


def create_app(config: Optional[APIConfig] = None):
    config = config or APIConfig()
    app = FastAPI()
    app.state.config = config
    app.mount("/public/images", StaticFiles(directory="public/images"), name="images")
    app.mount("/public", StaticFiles(directory="public"), name="public")
    @app.on_event('startup')
    async def startup():
        app.include_router(get_search_router(config))

    @app.on_event('shutdown')
    async def shutdown():
        pass

    return app

def main():
    app = create_app()
    uvicorn.run(app, port=8000, log_level="info")

if __name__ == "__main__":
    main()