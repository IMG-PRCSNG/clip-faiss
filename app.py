import enum
from pathlib import Path
from typing import List
import typer
from src.ioutils import write_dataset, read_dataset
from src.inference import setup_clip, AVAILABLE_MODELS

from src.search import (
    build_search_index,
    search_dataset,
)
from api import main

app = typer.Typer()


class _CLIPModel(str, enum.Enum):
    pass


CLIPModel = _CLIPModel("CLIPModel", {x: x for x in AVAILABLE_MODELS})


@app.command()
def extract_features(
    batch_size: int = typer.Option(
        1,
        help="Batch size that would fit your RAM/GPURAM",
    ),
    model: CLIPModel = typer.Option("ViT-B/32", help="CLIP Model to use"),  # type: ignore
    images_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory of images to index",
    ),
    save_to: Path = typer.Argument(
        ..., exists=False, help="File to write the extracted features to"
    ),
):
    if save_to.is_file():
        typer.confirm(
            "Are you sure you want to overwrite existing dataset?",
            abort=True,
            default=True,
        )
        print(f"Overwriting - {save_to}")

    files = sorted([x for x in images_dir.rglob("*") if x.is_file()])
    num_files = len(files)

    print(
        f'Processing: {num_files} files, ({[str(x) for x in files[:min(num_files, 10)]]}{"..." if num_files > 10 else ""})'
    )

    model_name = model.value

    extract_features, _ = setup_clip(model_name)

    features = extract_features(files, batch_size=batch_size)
    files_rel = [x.relative_to(images_dir) for x in files]
    write_dataset(save_to, features, files_rel, model_name)

    typer.Exit(0)


@app.command()
def search(
    dataset: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Extracted features .npy file",
    ),
    top_k: int = typer.Option(3, help="Top-k results to retrieve"),
    queries: List[str] = typer.Argument(..., help="Queries"),
):
    if len(queries) == 0:
        raise typer.BadParameter("Query cannot be empty")

    features, files, model_name = read_dataset(dataset)

    top_k = min(top_k, len(files))
    index = build_search_index(features)

    # Convert query to embedding
    _, extract_text_features = setup_clip(model_name)
    text_features = extract_text_features(queries)

    dist, ids = search_dataset(index, text_features, top_k=top_k)

    print(dist, [[Path(files[x]).name for x in top_ids] for top_ids in ids])


@app.command()
def serve():
    main()


if __name__ == "__main__":
    app()
