#!/usr/bin/env python3

from utils import set_verbose, vprint
from prob_matrix import ProbGraph
import heuristics as ht
import settings as st
import greedy as gd
import joblib
import random
import fire
import ic
import os

METHODS = {'gf': gd.greedy_filling,
           'ag': gd.alternating_greedy,
           'ts': gd.two_stage_greedy,
           'gb': gd.greedy_edge_block,
           'im': gd.greedy_im,
           'hc': ht.heur_centrality}

def run(matfile, S=None, k=1, repeat=1, algo='gf', fp=1e-3, rnd_fts=True, instance='', poly_fts=False,
        dist=False, corr=0.9, scaler='max_abs', alpha=0.2, measure='degree', B=20, R=1000, nv=5, cv=1, max_wks=0,
        key='capacity', one_ft=False, block=False, seed=None, output='.', spd_filter=0.01, mode=0, ub_mode=False,
        verbose=0):
    set_verbose(verbose)

    if not os.path.exists(str(matfile)):
        raise SystemExit('No data file!')

    # cache: [ [s1,...],... ]
    seeds_cache = (output or '.') + '/seeds.pkl'
    Ss = None
    if S is None or 'seeds.pkl' in str(S):
        path = S or seeds_cache
        if os.path.exists(path):
            vprint(f'Load seeds {path}...')
            Ss = joblib.load(path)
        Ss = Ss or [None] * repeat
    elif isinstance(S, (tuple, list)):
        if isinstance(S[0], int):
            Ss = [S]
        elif isinstance(S[0], (tuple, list)) and isinstance(S[0][0], int):
            Ss = S
    if Ss is None:
        raise SystemExit('Invalid seeds!')

    pg = ProbGraph(
        matfile, fp, rnd_fts, instance, poly_fts, dist,
        corr, scaler, key, one_ft, block, mode, verbose
    )

    st.set_inst(pg.G, B=B, R=R, nv=nv, cv=cv, block=block)
    ic.set_ub_mode(ub_mode)
    gd.set_max_wks(max_wks)

    if algo == 'im':
        res = METHODS[algo](pg.G, k)
        vprint(res)
        joblib.dump([res[0]], seeds_cache)
        return

    Xe = None
    if algo[:2] == 'gb':
        Xe = pg.xblock(pobj=1e-6, max_wks=0, bs=1000)

    random.seed(seed)

    spd_th = spd_filter * pg.G.vcount()
    vprint('Searching seeds...')
    for i, S in enumerate(Ss):
        if S is not None:
            continue
        spds = []
        for j in range(100):
            S = random.sample(pg.G.vs.indices, k)
            spd_pre, ub_pre = ic.influence(pg.G, S, st.R, 0)
            if spd_pre >= spd_th:
                Ss[i] = S
                break
            vprint(f'{S}: spread {spd_pre} < {spd_th}')
            spds.append(spd_pre)
        vprint(f'Trial: {j+1} times')
        if spd_pre < spd_th:
            raise SystemExit(f'Searching failed! Spread: max: {max(spds)}, mean: {sum(spds) / len(spds)}')

    if not os.path.exists(seeds_cache):
        joblib.dump(Ss, seeds_cache)

    results = []
    for i, S in enumerate(Ss):
        S = S[:k]
        vprint(f'Instance {i}, initial seeds: {S}')
        kargs = {'gb': (Xe, 0), 'gb.x': (Xe, 1), 'hc': measure, 'ag': alpha}
        res = METHODS[algo[:2]](pg.G, S, kargs.get(algo, None))
        vprint(res)
        results.append({'S': S, 'X': res[0], 'T': res[1], 'Tu': res[2], 'TX': (res[3] if len(res) >= 4 else None)})

    if results:
        joblib.dump(results, (output or '.') + '/results_%s.pkl' % (algo, measure)[algo=='hc'])


if __name__ == '__main__':
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire(run, serialize=lambda results: None)
