# CLIP + FAISS

Repo to put together OpenAI Clip and FAISS library together for a text based
image retrieval demo.

## Pre-requisites

(Tested on Debian 11 distro )

- conda

## Setup

Set up a conda environment and activate it

```bash
conda env create -f environment.yml
conda activate clip-faiss
```

## CLI

A Typer based CLI can be used to extract features and do a quick search

## Extract features

Use CLIP visual model to extract 512 embedding from images in a directory.
The features will be written an HDF5 File (with dataset 'features' with shape - N x 512, and filenames in attrs['files'])

```bash
python3 app.py extract-features IMAGE_DIR DATASET.h5
```

## Search

Query word embeddings are obtained from CLIP text model and searched against
the extracted features

```bash
python3 app.py search --dataset DATASET.h5 QUERY1 QUERY2 [...]
```