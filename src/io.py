import typing
from typing import Literal, Optional, Generator, Tuple
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import h5py


@contextmanager
def get_dataset(
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


def write_array_to_dataset(dataset: h5py.Dataset, arr: np.ndarray):
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
