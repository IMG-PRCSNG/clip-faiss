import itertools
from pathlib import Path
from typing import Generator, List, Union
from PIL import Image
import numpy as np
import torch
import clip

IS_CUDA = torch.cuda.is_available()

AVAILABLE_MODELS = clip.available_models()


def _load_clip(model_name: str):

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model - {model_name}, available models: {AVAILABLE_MODELS}"
        )
    print(f"Loading CLIP (model: {model_name})...")
    model, preprocess = clip.load(model_name)

    if IS_CUDA:
        model.cuda()
    model.eval()
    print("Loaded")

    return model, preprocess


def batched(iterable, n: int):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            return
        yield batch


def setup_clip(model_name: str = "ViT-B/32"):

    model, preprocess = _load_clip(model_name)
    input_dim = model.visual.input_resolution
    mean_tensor = (
        torch.Tensor([0.48145466, 0.4578275, 0.40821073])
        .reshape(3, 1, 1)
        .repeat(1, input_dim, input_dim)
    )

    def _preprocess(p: Path):
        try:
            with Image.open(p) as im:
                return preprocess(im)
        except Exception as e:
            print(f"warning: failed to process {p} - {e}")
            return mean_tensor

    def extract_image_features(
        images: Union[Path, List[Path]], batch_size: int = 1
    ) -> Generator[np.ndarray, None, None]:
        _files = [images] if isinstance(images, Path) else images
        with torch.no_grad():
            for batch in batched(_files, batch_size):
                _input = torch.stack([_preprocess(p) for p in batch], dim=0)

                if IS_CUDA:
                    _input = _input.cuda()

                output = model.encode_image(_input).float()
                output /= torch.linalg.norm(output, dim=-1, keepdims=True)

                yield output.cpu().numpy()

    def extract_text_features(
        queries: List[str],
    ) -> np.ndarray:
        text_tokens = clip.tokenize([f"This is a photo of a {x}" for x in queries])
        if IS_CUDA:
            text_tokens = text_tokens.cuda()

        with torch.no_grad():
            output = model.encode_text(text_tokens).float()
            output /= torch.linalg.norm(output, dim=-1, keepdims=True)

            return output.cpu().numpy()

    return extract_image_features, extract_text_features
