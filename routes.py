from functools import partial
from pathlib import Path
from typing import Dict, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, validator

from config import APIConfig

from src.schemas import ImageInfo
from src.ioutils import read_dataset, get_image_info
from src.inference import setup_clip
from src.search import build_search_index, search_dataset


def get_search_router(config: APIConfig):
    router = APIRouter(
        tags=["search"],
    )

    class SearchResponse(BaseModel):
        link: str
        thumbnail: str
        distance: float
        info: ImageInfo

        @validator("distance")
        def round_distance(cls, v):
            return round(v, config.precision)

    features, files, model_name = read_dataset(config.dataset)
    images_dir = config.images_dir

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
                    thumbnail=f"thumbs/{files[_id]}",
                    link=f"images/{files[_id]}",
                    distance=dist[qid][kid],
                    info=get_image_info(images_dir / files[_id], images_dir),
                )
                for kid, _id in enumerate(ids[qid])
            ]
            for qid in range(len(q))
        }
        return response

    return router
