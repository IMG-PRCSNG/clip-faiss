from functools import partial
from pathlib import Path
from typing import Dict, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, validator

from config import APIConfig

from src.io import read_dataset
from src.inference import setup_clip
from src.search import build_search_index, search_dataset


def get_search_router(config: APIConfig):
    router = APIRouter(
        tags=["search"],
    )

    class SearchResponse(BaseModel):
        link: str
        name: str
        distance: float

        @validator("distance")
        def round_distance(cls, v):
            return round(v, config.precision)

    features, files, model_name = read_dataset(config.dataset)
    index = build_search_index(features)

    num_files = len(files)
    _, extract_text_features = setup_clip(model_name)

    search_fn = partial(search_dataset, index)

    @router.get("/search", response_model=Dict[str, List[SearchResponse]])
    async def search(
        q: List[str] = Query(
            default=[],
        ),
        top_k: int = Query(config.top_k, gt=0, le=min(25, num_files)),
    ):
        if len(q) == 0:
            raise HTTPException(
                400, {"message": "Must be called with search query term"}
            )

        text_features = extract_text_features(q)
        dist, ids = search_fn(text_features, top_k=top_k)

        response = {
            q[qid]: [
                SearchResponse(
                    name=Path(files[_id]).name,
                    distance=dist[qid][kid],
                    link=f"/public/{files[_id]}",
                )
                for kid, _id in enumerate(ids[qid])
            ]
            for qid in range(len(q))
        }
        return response

    return router
