from contextlib import contextmanager
import itertools
from pathlib import Path
from typing import Generator, List, Literal, Optional, Union
import typing
import numpy as np
from PIL import Image
import clip
import torch
import faiss
import typer
import h5py
from tqdm import tqdm

IS_CUDA = torch.cuda.is_available()

app = typer.Typer()

@contextmanager
def get_dataset(path: Path, mode: Literal['r', 'w'] = 'r', n_dim: Optional[int] = None) -> Generator[h5py.Dataset, None, None]:
    with h5py.File(path, mode) as f:
        if mode != 'r':
            # Write enabled
            if n_dim is None:
                raise ValueError('n_dim cannot be None in write mode')
            if 'features' not in f:
                f.create_dataset('features', shape=(0, n_dim), maxshape=(None, n_dim))

        yield typing.cast(h5py.Dataset, f['features'])

def write_array_to_dataset(dataset: h5py.Dataset, arr: np.ndarray):
    shape_diff = (len(dataset.shape) - len(arr.shape))
    if shape_diff > 1:
        raise ValueError(f"Unsupported array shape - must be {dataset.shape[1:]} or {('N',) + dataset.shape[1:]}")

    iarr = arr
    if shape_diff == 1:
        iarr = np.expand_dims(arr, axis=0)

    n = dataset.shape[0]
    b = iarr.shape[0]

    # Resize
    dataset.resize(n + b, axis=0)

    # Write
    dataset[-b:, ...] = iarr

def _load_clip():
    model, preprocess = clip.load("ViT-B/32")

    if IS_CUDA:
        model.cuda()
    model.eval()

    return model, preprocess

def batched(iterable, n:int):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            return
        yield batch

def setup():

    print('Loading CLIP...')
    model, preprocess = _load_clip()
    print('Loaded')

    def _preprocess(p: Path):
        with Image.open(p) as im:
            return preprocess(im.convert('RGB'))

    def extract_image_features(images: Union[Path, List[Path]], batch_size:int = 1):
        _files = [images] if isinstance(images, Path) else images
        with torch.no_grad():
            for batch in batched(_files, batch_size):
                _input = torch.stack([_preprocess(p) for p in batch], dim=0)

                if IS_CUDA:
                    _input = _input.cuda()

                yield model.encode_image(_input).float()

    def extract_text_features(queries: List[str]):
        text_tokens = clip.tokenize([f"This is a photo of a {x}" for x in queries])
        if IS_CUDA:
            text_tokens = text_tokens.cuda()

        with torch.no_grad():
            return model.encode_text(text_tokens).float()

    return extract_image_features, extract_text_features

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

    extract_features, _ = setup()

    features = extract_features(files, batch_size=batch_size)
    normalized = (x / torch.linalg.norm(x, dim=-1, keepdims=True) for x in features)

    with get_dataset(save_to, 'w', n_dim=512) as ds, tqdm(total=num_files) as pbar:
        ds.attrs['files'] = [f'{x}' for x in files]
        for arr in normalized:
            write_array_to_dataset(ds, arr.cpu().numpy())
            print(arr.shape, ds.shape)
            pbar.update(arr.shape[0])

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
    n_dim = -1
    _, extract_text_features = setup()

    with get_dataset(dataset, 'r') as ds:
        files = ds.attrs['files']
        n_dim = ds.shape[-1]
        index = faiss.IndexFlatIP(n_dim)
        index.add(ds[:, ...])

    # Convert query to embedding
    text_features = extract_text_features(queries)
    text_features /= torch.linalg.norm(text_features, dim=-1, keepdims=True)

    text_features_np = text_features.cpu().numpy()
    dist, ids = index.search(x=text_features_np, k=top_k)

    print(dist, [[Path(files[x]).stem for x in top_ids]  for top_ids in ids ])

if __name__ == '__main__':
    app()