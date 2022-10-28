from functools import partial
from pathlib import Path
from typing import Dict, List
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, validator
from config import APIConfig

from src.inference import setup_clip
from src.search import search_dataset, setup_index

def get_search_router(config: APIConfig):
    router = APIRouter(
        tags=["search"],
    )
    class SearchResponse(BaseModel):
        link: str
        name: str
        distance: float

        @validator('distance')
        def round_distance(cls, v):
            return round(v, config.precision)

    index, files = setup_index(config.dataset)
    num_files = len(files)
    _, extract_text_features = setup_clip()

    search_fn = partial(search_dataset, index, extract_text_features)

    @router.get('/search', response_model=Dict[str, List[SearchResponse]])
    async def search(q: List[str] = Query(default=[],), top_k: int = Query(config.top_k, gt=0)):
        if len(q) == 0:
           raise HTTPException(400, { 'message': 'Must be called with search query term'})

        dist, ids = search_fn(q, top_k=min(top_k, num_files))
        dist = np.around(dist, decimals = config.precision)
        response = {
            q[qid]: [
                SearchResponse(
                    name=Path(files[_id]).name,
                    distance=dist[qid][kid],
                    link=f'/public/{files[_id]}'
                ) for kid, _id in enumerate(ids[qid])
            ] for qid in range(len(q))
        }
        print(response)
        return response
    return router
