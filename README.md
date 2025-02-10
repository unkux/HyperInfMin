# HyperInfMin

Hyperparametric Influence Minimization.

## Environment

Tested with `Python 3.10.10` on `Linux`.

## Instruction

The main file is `run.py`. `ic` module (in `pyx`) needs to be compiled in advance.

```
NAME
    run.py

SYNOPSIS
    run.py MATFILE <flags>

POSITIONAL ARGUMENTS
    MATFILE
    Input probability matrix or hyperparameters.

FLAGS
    -S, --S=S
        Type: Optional[]
        Default: None
        Seed nodes.
    --k=K
        Default: 1
        Number of seeds.
    --spd_filter=SPD_FILTER
        Default: 0.01
        Spread filter for generating random seeds.
    --repeat=REPEAT
        Default: 1
        Instance repetition.
    --algo=ALGO
        Default: 'gf'
        Algorithms: [gf, ts, ag, gb, gb.x, hc].
    --measure=MEASURE
        Default: 'degree'
        Centrality measures: [betweenness, closeness, degree, eigenvector, pagerank].
    --mode=MODE
        Default: 0
        Influence estimation approaches: [Monte Carlo, Steady-State Spread].
    -u, --ub_mode=UB_MODE
        Default: False
        Run algorithms in upper bound mode.
    --alpha=ALPHA
        Default: 0.2
        Parameter for AG, initial exploration ratio.
    -B, --B=B
        Default: 20
        Budget for feature intervention.
    -n, --nv=NV
        Default: 5
        Feature intervention level.
    --cv=CV
        Default: 1
        Unit cost of feature intervention.
    -R, --R=R
        Default: 1000
        Monte Carlo running times for IC model.
    --max_wks=MAX_WKS
        Default: 0
        Run parallel with multicore CPU.
    -i, --instance=INSTANCE
        Default: ''
        Instance containing feature data.
    -f, --fp=FP
        Default: 0.001
        Influence probability filter for input graph.
    --key=KEY
        Default: 'capacity'
        Feature for intervention.
    --rnd_fts=RND_FTS
        Default: True
        Random feature [features.py].
    -p, --poly_fts=POLY_FTS
        Default: False
        Polynomial feature [features.py].
    -d, --dist=DIST
        Default: False
        Distance feature [features.py].
    --corr=CORR
        Default: 0.9
        Feature correlation [features.py].
    --scaler=SCALER
        Default: 'max_abs'
        Feature scaler [features.py].
    --one_ft=ONE_FT
        Default: False
        Use one feature.
    -b, --block=BLOCK
        Default: False
        Blocking test.
    --seed=SEED
        Type: Optional[]
        Default: None
        Random seed number.
    --output=OUTPUT
        Default: '.'
        Output directory.
    -v, --verbose=VERBOSE
        Default: 0
```

## Dataset

[Power Grids](https://github.com/unkux/Learn_CF) and [Social Networks](https://snap.stanford.edu/data/ego-Facebook.html).
