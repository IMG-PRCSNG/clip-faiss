import itertools
from pathlib import Path
from typing import List, Union
from PIL import Image
import torch
import clip

IS_CUDA = torch.cuda.is_available()

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

def setup_clip():

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