import typing
from typing import List, Literal, Optional, Generator, Tuple, cast
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import h5py
from PIL import Image, IptcImagePlugin
from tqdm import tqdm

from .schemas import ImageInfo


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
    """
    Features is a 2D array of shape (len(file_names), n_dim)
    """

    # Read first array from generator to get shape
    arr = next(features)
    n_dim = arr.shape[-1]

    with _get_dataset(dataset, "w", n_dim=n_dim) as (ds, fs), tqdm(
        total=len(file_names)
    ) as pbar:

        ds.attrs["model"] = model_name

        # Write the first array
        _write_array_to_dataset(ds, arr)
        pbar.update(arr.shape[0])

        # Write remaining arrays
        for _arr in features:
            _write_array_to_dataset(ds, _arr)
            pbar.update(_arr.shape[0])

        print("writing file names")
        _write_array_to_dataset(fs, np.array([str(x) for x in file_names]))

        print(f"Done - wrote features ({model_name}) with shape: {ds.shape}")

    pass


ENCODING_MAP = {
    "CP_1252": "cp1252",
    "CP_UTF8": "utf-8",
}


def get_image_info(p: Path, basedir: Optional[Path] = None):
    # See https://gist.github.com/bhaskarkc/abcbc4a35229815bd6ce4ab7372748f9

    with Image.open(p) as im:
        w, h = im.size
        iminfo = ImageInfo(
            filename=str(p.relative_to(basedir) if basedir else p),
            width=w,
            height=h,
            title=p.name,
        )

        # Add iptc
        iptc = IptcImagePlugin.getiptcinfo(im)
        if not iptc:
            return iminfo

        encoding = iptc.get((2, 183), b"utf-8").decode()

        encoding = ENCODING_MAP.get(encoding, "utf-8")

        iminfo.title = iptc.get((2, 85), iminfo.title.encode()).decode(encoding)
        iminfo.copyright = iptc.get((2, 116), b"").decode(encoding)
        iminfo.caption = iptc.get((2, 120), b"").decode(encoding)

        return iminfo
