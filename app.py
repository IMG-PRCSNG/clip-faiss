from pathlib import Path
from typing import List
import torch
import typer
from tqdm import tqdm
import numpy as np
from src.inference import setup_clip

from src.io import get_dataset, write_array_to_dataset
from src.search import search_dataset, setup_index
from api import main

app = typer.Typer()

@app.command()
def extract_features(
    batch_size: int = typer.Option(
        1,
        help="Batch size that would fit your RAM/GPURAM",
    ),
    images_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory of images to index"
    ),
    save_to: Path = typer.Argument(
        ...,
        exists=False,
        help="File to write the extracted features to"
    )
):
    if save_to.is_file():
        typer.confirm('Are you sure you want to overwrite existing dataset?', abort=True, default=True)
        print(f'Overwriting - {save_to}')

    files = sorted([x for x in images_dir.rglob('*') if x.is_file()])
    num_files = len(files)

    print(f'Processing: {num_files} files, ({[str(x) for x in files[:min(num_files, 10)]]}{"..." if num_files > 10 else ""})')

    extract_features, _ = setup_clip()

    features = extract_features(files, batch_size=batch_size)
    normalized = (x / torch.linalg.norm(x, dim=-1, keepdims=True) for x in features)

    with get_dataset(save_to, 'w', n_dim=512) as (ds, fs), tqdm(total=num_files) as pbar:

        for arr in normalized:
            write_array_to_dataset(ds, arr.cpu().numpy())
            pbar.update(arr.shape[0])

        print(ds.shape)

        print('writing file names')
        fs.resize(num_files, axis=0)
        fs[:, ...] = np.array([str(x) for x in files])

        print('Done')

    typer.Exit(0)

@app.command()
def search(
    dataset: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Extracted features .npy file"
    ),
    queries: List[str] = typer.Argument(
        ...,
        help="Queries"
    )
):
    if len(queries) == 0:
        raise typer.BadParameter('Query cannot be empty')

    index = None
    top_k = 3

    index, files = setup_index(dataset)
    _, extract_text_features = setup_clip()
    dist, ids = search_dataset(index, extract_text_features, queries, top_k=top_k)

    print(dist, [[Path(files[x]).stem for x in top_ids]  for top_ids in ids ])

@app.command()
def serve():
    main()

if __name__ == '__main__':
    app()