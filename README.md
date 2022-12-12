# CLIP + FAISS

Repo to put together OpenAI Clip and FAISS library together for a text based
image retrieval demo.

## Pre-requisites

(Tested on Debian 11 distro )

- conda (You can install miniconda - [instructions](https://docs.conda.io/en/latest/miniconda.html))

## Setup

Set up a conda environment and activate it

```bash
conda env create -f environment.yml
conda activate clip-faiss
```

## CLI

A Typer based CLI can be used to extract features and do a quick search

### Extract features

Use CLIP visual model to extract 512 embedding from images in a directory.
The features will be written an HDF5 File (with dataset 'features' with shape - N x 512, and filenames in attrs['files'])

```bash
python3 app.py extract-features IMAGE_DIR DATASET.h5
```

### Search

Query word embeddings are obtained from CLIP text model and searched against
the extracted features

```bash
python3 app.py search --dataset DATASET.h5 QUERY1 QUERY2 [...]
```

## Web Demo

A FAST API + HTML / CSS / JS app to make queries against the extracted features.

### Serve

```bash
ln -s IMAGES_DIR public/images
# Optionally, you can also create thumbnails and set it accordingly or re-use image dir
ln -s IMAGES_DIR public/thumbs

DATASET="DATASET.h5" python3 app.py serve
```

You can now open http://localhost:8000/public/index.html to view the demo.
