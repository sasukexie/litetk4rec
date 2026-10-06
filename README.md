# LiteTK4Rec

Code for the paper **"LiteTK4Rec: A Lightweight Attention Time-Kernel with Computable Criteria
for When Temporal Modeling Helps Sequential Recommendation"**.

## Layout

| Path | What it is |
|---|---|
| `recbole/model/sequential_recommender/litetk4rec.py` | the LiteTK4Rec model (a learnable temporal bias added to the attention scores) |
| `recbole/properties/model/LiteTK4Rec.yaml` | model defaults |
| `recbole/` | the rest of the codebase and the baselines, built on [RecBole](https://github.com/RUCAIBox/RecBole) (MIT) |
| `run_main.py` | entry point (argparse) |
| `run_base.py`, `run_hyper.py` | baseline runs and the hyper-parameter search driver |
| `dataset/` | processed interaction files for the datasets used in the paper |
| `common/`, `zone/` | helper utilities and the run configurations used for the reported results |

## Environment

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended. Paths in the configuration files are relative to this directory.

## Running

```bash
python run_main.py --model LiteTK4Rec --dataset ml-3m
```

The temporal parameterization is selected by `temporal_type` in the model configuration:

* `none` - no time term. This is the no-time baseline; at initialization it is bit-for-bit
  identical to SASRec, which is what makes every reported gain attributable to time alone.
* `interval` - a TiSASRec-style interval embedding (input-side injection).
* `session3` - the head-wise interval table of the paper (261 parameters at `H=8, K=32`).
* `ckernel` - the continuous time kernel of the paper (95 parameters).

## Datasets

`dataset/` bundles the ten criterion datasets used in the paper (Amazon_Books, BeerAdvocate,
diginetica, gowalla, ml-3m, steam, netflix, mind, lfm1b-tracks, foursquare_TKY), in the processed
RecBole format used here. The ladder and held-out sets (ml-1m, yelp1m, RentTheRunway, ml-100k,
movielens) are public benchmarks obtained from their original sources; the complete processed set
is also available at the download url given in `dataset/README.md`. Please cite the original
source of every dataset you use.


## Reproducing the reported numbers

Every number in the paper is tied to a specific run. The Supplementary Material of the paper gives,
for each individual cell, the run it comes from (dataset, configuration, number of epochs and seed).
The criteria of the paper are computed from the interaction log alone by a single linear scan, as
described in Section 3 of the paper.

## License

`recbole/` is derived from RecBole, which is released under the MIT License (see `LICENSE`).
The modifications made for this work are released under the same license.
