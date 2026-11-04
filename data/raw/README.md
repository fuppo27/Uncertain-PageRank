Raw data as obtained. Not under version control by default (`.gitignore` excludes it and
tracks only `README.md` and `.gitkeep`); the public release additionally tracks the two
laboratory networks, as `DATA.md` explains.

Raw data is never rewritten in place. Converted data goes to `data/interim/` and final
versions to `data/processed/`.

## Layout

```
data/raw/
├── original/          Laboratory data used in the conference version (IEEE BigData 2020)
│   ├── follow/        Ameba Blog follower network    (56,604 nodes / 734,737 links)
│   └── trackback/     goo Blog trackback network     (12,047 nodes /  53,315 links)
└── snap/              Public data from SNAP (placed by scripts/download_snap.py)
```

The conference version also used an Enron email network, which this work does not; it is
not part of this repository.

The file format under `original/` has changed since the conference version, so the reader
in `src/uncertain_pagerank/datasets.py` was written against the files as they are now.
Compressed files are fine (`.gz`, `.zip` and `.tar.gz` are all accepted).

## SNAP data

`scripts/download_snap.py` fetches each dataset and records its URL, byte count, SHA256
and retrieval time in `manifest.json`. That file is version-controlled for reproducibility;
the data itself is not, and is not redistributed here.
