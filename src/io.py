import typing
from typing import List, Literal, Optional, Generator, Tuple, cast
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm


@contextmanager
def _get_dataset(
    path: Path, mode: Literal["r", "w"] = "r", n_dim: Optional[int] = None
) -> Generator[Tuple[h5py.Dataset, h5py.Dataset], None, None]:
    with h5py.File(path, mode) as f:
        if mode != "r":
            # Write enabled
            if n_dim is None:
                raise ValueError("n_dim cannot be None in write mode")
            if "features" not in f:
                f.create_dataset("features", shape=(0, n_dim), maxshape=(None, n_dim))

            if "files" not in f:
                dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset("files", shape=(0,), dtype=dt, maxshape=(None,))

        yield typing.cast(h5py.Dataset, f["features"]), typing.cast(
            h5py.Dataset, f["files"]
        )


def _write_array_to_dataset(dataset: h5py.Dataset, arr: np.ndarray):
    shape_diff = len(dataset.shape) - len(arr.shape)
    if shape_diff > 1:
        raise ValueError(
            f"Unsupported array shape - must be {dataset.shape[1:]} or {('N',) + dataset.shape[1:]}"
        )

    iarr = arr
    if shape_diff == 1:
        iarr = np.expand_dims(arr, axis=0)

    n = dataset.shape[0]
    b = iarr.shape[0]

    # Resize
    dataset.resize(n + b, axis=0)

    # Write
    dataset[-b:, ...] = iarr


def read_dataset(dataset: Path) -> Tuple[np.ndarray, List[str], str]:
    with _get_dataset(dataset, "r") as (ds, fs):
        model_name: str = cast(str, ds.attrs["model"])
        files = [x.decode("utf-8") for x in fs[:]]
        features: np.ndarray = ds[:, ...]
        print(f"Read ({features.shape}) features (model: {model_name})")
        return features, files, model_name


def write_dataset(
    dataset: Path,
    features: Generator[np.ndarray, None, None],
    file_names: List[Path],
    model_name: str,
):
    with _get_dataset(dataset, "w", n_dim=512) as (ds, fs), tqdm(
        total=len(file_names)
    ) as pbar:

        ds.attrs["model"] = model_name

        for arr in features:
            _write_array_to_dataset(ds, arr)
            pbar.update(arr.shape[0])

        print("writing file names")
        _write_array_to_dataset(fs, np.array([str(x) for x in file_names]))

        print(f"Done - wrote features ({model_name}) with shape: {ds.shape}")

    pass
