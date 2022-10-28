from pathlib import Path
from typing import List
from src.io import get_dataset
import torch
import faiss

def setup_index(dataset: Path):
    global index, files
    with get_dataset(dataset, 'r') as ds:
        files = ds.attrs['files']
        n_dim = ds.shape[-1]
        index = faiss.IndexFlatIP(n_dim)
        index.add(ds[:, ...])

        return index, files

def search_dataset(index, extract_text_features, queries: List[str], top_k:int = 3):
    # Convert query to embedding
    text_features = extract_text_features(queries)
    text_features /= torch.linalg.norm(text_features, dim=-1, keepdims=True)

    text_features_np = text_features.cpu().numpy()
    dist, ids = index.search(x=text_features_np, k=top_k)

    return dist, ids