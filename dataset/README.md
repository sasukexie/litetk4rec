# Datasets

The files here are the processed interaction files used in the paper. All datasets are public
benchmarks; please cite their original sources and observe their own licences.

## What is bundled

The ten criterion datasets, in exactly the processed format used in the paper (RecBole
interaction format, one `user item timestamp` record per line):

    Amazon_Books, BeerAdvocate, diginetica, gowalla, ml-3m, steam,
    netflix, mind, lfm1b-tracks, foursquare_TKY

The ladder and held-out sets (ml-1m, yelp1m, RentTheRunway, ml-100k, movielens) are not bundled
here; they are public and are obtained from the sources listed below.

## Download the complete set

All datasets used in the paper, in exactly the processed format used here:

**Download url:** https://drive.google.com/drive/folders/1CmuRr9HHVfkmtxaJScjjGDw-Gcon8NF5?usp=drive_link

## Sources


| Dataset | Original source |
|---|---|
| ml-1m, ml-100k, movielens | GroupLens Research, MovieLens (https://grouplens.org/datasets/movielens/) |
| ml-3m | a larger MovieLens-family slice, processed the same way as ml-1m |
| netflix | Netflix Prize data (Netflix, Inc.; see the terms of use of the original release) |
| mind | Microsoft News Dataset (MIND), https://msnews.github.io/ (research licence) |
| lfm1b-tracks | LastFM-1B dataset (request-only) |
| BeerAdvocate, Amazon_Books, steam | the review datasets collected by J. McAuley and co-authors |
| gowalla | SNAP, Stanford University, https://snap.stanford.edu/data/loc-gowalla.html |
| foursquare_TKY | the Tokyo slice of the Foursquare global check-in dataset (request-only) |
| diginetica | CIKM Cup 2016 challenge data |
| yelp1m | the Yelp Open Dataset |
| RentTheRunway | the public RentTheRunway recommendation dataset |

## Format

The files follow the interaction format used by RecBole: `<name>.inter` holds one
`user item timestamp` record per line, with `<name>.item` / `<name>.user` for item and user features
where available. Some directories also keep the original compressed file (`<name>.inter.gz`).
