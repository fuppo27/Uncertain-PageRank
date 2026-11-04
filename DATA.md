# Data

## What is here

`outputs/` contains the result CSVs behind every table and figure in the paper, so the
reported numbers can be checked without re-running the experiments.

`data/raw/snap/manifest.json` records the URL, byte count, SHA-256 and retrieval time of
each public dataset used. The datasets themselves are **not redistributed**; run
`scripts/download_snap.py` to fetch them from the Stanford Large Network Dataset
Collection and verify the checksums.

## What is not here, and why

Two networks used in the paper are held by our laboratory and are not public data:

| Name in the paper | Source |
|---|---|
| Follow NW | follower relations crawled from a blog service |
| Trackback NW | trackback relations crawled from a blog service |

We include them under `data/raw/original/`, in the `.dlbl` format described in
`data/raw/README.md`. The files contain adjacency structure only: a node is identified
by its line number, link targets are the corresponding line numbers, and no account
name, URL or mapping table is present. They were collected by our laboratory for the
conference version of this work.

## Effect on reproducibility

All enumeration-scale and medium-scale results use synthetic generators that are
reproduced exactly by the code here, from the seeds in `configs/experiment/`.

Of the real-network results, the per-source validation is reported on wiki-Vote and
Trackback NW, and the cost measurements additionally use soc-Epinions1 and Follow NW.
With the laboratory data included, every real-network result in the paper can be
reproduced from this repository once the public datasets are downloaded.

## Link probabilities

No network studied here ships with link probabilities. We assign them synthetically, as
stated in the paper, and the assignment is part of the experiment configuration rather
than of the data. `scripts/q_sensitivity.py` reports how much the conclusions depend on
that choice.

## Provenance recorded with each result

Each directory under `outputs/` carries a `meta.json` with the time, the Python, NumPy
and SciPy versions, the machine, and the experiment configuration in full. Two fields
need a word of explanation.

`git.commit` and `git.branch` refer to the private development repository the
experiments were run from, not to this one, so they cannot be resolved here. The code in
this repository is what produced the results; the commit is recorded only to tie a run
to a point in that history.

`git.dirty` is `"yes"` throughout: the runs were made from a working tree that had
uncommitted edits, which is the normal state while an experiment is being developed. It
does not mean the code here differs from the code that ran. We re-derived the figures
and the budget curve from the files in this repository as a check, and they reproduce
the values in the paper.
