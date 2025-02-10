#!/usr/bin/env python3

from itertools import chain, combinations_with_replacement
from utils import set_verbose, vprint, clip_max
from bounded_pool import BoundedPool
from scipy import sparse, io
from ic import init_repr
from glob import glob
import settings as st
import igraph as ig
import numpy as np
import features
import fire
import os
import re

def prob_matrix(F, D, theta, G=None, on_matrix=True):
    F_SIZE, F2_SIZE = F.shape[1], 2*F.shape[1]
    Ft1, Ft2 = F.dot(theta[:F_SIZE]), F.dot(theta[F_SIZE:F2_SIZE])
    if D.ndim == 3:
        Dw = np.sum(theta[F2_SIZE:, None, None] * D, axis=0)
    else:
        Dw = D * theta[-1]
    if on_matrix and Dw.ndim == 2:
        # arr: theta^T x_{uv}
        M = np.add.outer(Ft1, Ft2) + Dw
        clip_max(M)
        P = 1/(1+np.exp(-M))
        np.fill_diagonal(P, 0)
    elif G is not None:
        P = np.zeros(len(G.es))
        for edge in G.es:
            (u, v), i = edge.tuple, edge.index
            d = Dw[u, v] if Dw.ndim == 2 else Dw[i]
            P[i] = 1/(1+np.exp(-(Ft1[u]+Ft2[v]+d)))
    else:
        P = None
    return P

def rnd_prob_graph(matfile, rnd_fts=True, show=False):
    G = ig.Graph.Read_Edgelist(matfile, directed=True)
    rg_file = (os.path.dirname(matfile) or '.') + '/rnd_graph.npz'
    F, theta = None, None
    if os.path.exists(rg_file):
        vprint('Load random graph...')
        with np.load(rg_file) as data:
            if 'P' in data:
                P = data['P']
            else:
                F, D, theta = (data[f] for f in ['F', 'D', 'theta'])
                P = prob_matrix(F, D, theta, G, on_matrix=False)
    else:
        vprint('Build random graph...')
        if rnd_fts:
            # NOTE: use one feature, other values are encoded in D
            F, D = np.random.rand(len(G.vs), 1), np.random.normal(10, 10, len(G.es))
            # NOTE: ensure probability is monetone decresing
            theta = -np.random.rand(2*F.shape[1] + 1)
            np.savez_compressed(rg_file, F=F, D=D, theta=theta)
            P = prob_matrix(F, D, theta, G, on_matrix=False)
        else:
            P = np.random.rand(len(G.es))
            np.savez_compressed(rg_file, P=P)
    if show:
        vprint(f"Probability: (range: [{min(P)}, {max(P)}], mean: {np.mean(P)})\n")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.hist(P, bins=50, weights=np.ones_like(P)/len(P), color="skyblue", edgecolor="black", alpha=0.7)
        plt.show()
        raise SystemExit()
    G.es['weight'] = P.tolist()
    return G, F, theta

def load_data(matfile, rnd_fts=True, instance='', poly_fts=False, dist=False, corr=0.9, scaler='max_abs', verbose=0):
    if not os.path.exists(str(matfile)):
        return None
    if '.mtx' in matfile:
        P = io.mmread(matfile).toarray()
        return P
    elif '.npy' in matfile:
        mat = np.load(matfile)
        if os.path.exists(str(instance)):
            F, D, scaler = features.build(instance, poly_fts=poly_fts, corr=corr, scaler=scaler, verbose=verbose)
            D0 = np.ones((1, F.shape[0], F.shape[0]))
            D = np.concatenate((D, D0)) if dist else D0
            P = prob_matrix(F, D, mat)
            return P, F, mat, scaler
        elif mat.ndim == 2 and mat.shape[0] == mat.shape[1] and mat.max() <= 1 and mat.min() >= 0:
            return mat
    elif '.edges' in matfile:
        return rnd_prob_graph(matfile, rnd_fts)
    return None

# get the relevant polynomial features
def get_ft_ind(key='capacity', order=2, bad_fts_id=[], one_ft=False):
    ft_ind = None
    if key in features.FEATURES:
        k = features.FEATURES.index(key)
        ft_size = len(features.FEATURES)
        comb = list(chain.from_iterable(combinations_with_replacement(range(ft_size), i) for i in range(1, order+1)))
        ft_list = [i for i in range(len(comb)) if i not in bad_fts_id]
        ft_ind = {i: comb[ic].count(k) for i, ic in enumerate(ft_list) if k in comb[ic]}
        if one_ft and ft_ind:
            k = sorted(ft_ind)[0]
            ft_ind = {k: ft_ind[k]}
    return ft_ind

_update = None
def upd_graph(G, X, A):
    if _update is None:
        raise SystemExit('Not instantiated!')
    return _update(G, X, A)

def search(f, xrange, fobj=1e-6, eps=1e-7, xint=True):
    x0, x1 = xrange
    y0, y1 = f(x0), f(x1)
    #print(y0, y1)
    if y0 < fobj - eps:
        return 0
    if y1 > fobj + eps:
        return -1
    for x, y in [(x0, y0), (x1, y1)]:
        if abs(y - fobj) <= eps:
            return x
    xp = 0
    for i in range(30):
        x = (x0 + x1) / 2
        if xint:
            x = round(x)
        y = f(x)
        #print(i, (x0, x1), x, y)
        if abs(y - fobj) <= eps:
            return x
        elif y > fobj + eps:
            x0 = x
        else:
            x1 = x
        if abs(x - xp) <= eps:
            return (x, x1)[xint]
        xp = x

def _search_pb(indices, Vb, Pb, Fb, fvals, pobj):
    Xe = {}
    for k, v, p, t in zip(indices, Vb, Pb, Fb):
        if p > pobj:
            x = search(
                lambda x: 1/(1+(1/p-1)*np.exp(-np.sum(t*((x+1)**fvals-1)))),
                xrange=(0, st.N[v]), fobj=pobj, eps=1e-7, xint=True
            )
            if x != 0:
                Xe[k] = x
    return Xe

# NOTE: load F theta, compute initial G, update G given X
class ProbGraph:
    def __init__(self, matfile, fp=1e-3, rnd_fts=True, instance='', poly_fts=False, dist=False,
                 corr=0.9, scaler='max_abs', key='capacity', one_ft=False, block=False, mode=0, verbose=0):
        set_verbose(verbose)
        self.G, self.P, self.F, self.theta = [None] * 4
        self.poly_fts, self.scaler = poly_fts, None
        data = load_data(matfile, rnd_fts, instance, poly_fts, dist, corr, scaler, verbose)
        if isinstance(data, tuple):
            if isinstance(data[0], np.ndarray):
                self.P, self.F, self.theta, self.scaler = data
                ptn = (os.path.dirname(matfile) or '.') + f'/{st.F_PMAT}'
                if not glob(ptn % '*'):
                    self.P[self.P<1e-3] = 0
                    vprint('Save probability matrix')
                    label = ''.join(re.findall(st.F_THETA % '(.*)', os.path.basename(matfile))[:1])
                    io.mmwrite(ptn % f'l_{label}_1e-3', sparse.csr_matrix(self.P))
            else:  # if isinstance(data[0], ig.Graph):
                self.G, self.F, self.theta = data
                vprint(self.G.summary())
        elif isinstance(data, np.ndarray):
            self.P = data
        else:
            raise SystemExit('No valid data!')

        if self.theta is not None:
            vprint(f"Parameters: (dim: {len(self.theta)}, range: {min(self.theta), max(self.theta)}, mean: {self.theta.mean()})\n")

        if self.G is not None:
            edge_rm = [e.index for e in self.G.es if e['weight'] < fp]
            self.G.delete_edges(edge_rm)
        else:
            self.P[self.P<fp] = 0
            self.G = ig.Graph.Weighted_Adjacency(self.P.tolist(), mode=ig.ADJ_DIRECTED)

        P = self.G.es['weight']
        vprint(f"Probability graph: (nodes: {len(self.G.vs)}, edges: {len(P)}, range: [{min(P)}, {max(P)}], mean: {np.mean(P)})\n")

        init_repr(self.G, mode)

        self.upd_param(key, one_ft, block)
        self.FT = None
        if self.F is not None:
            fkeys, fvals = self.ft_pair
            self.FT = self.F[:, fkeys] * self.theta[self.F.shape[1]:][fkeys]

        global _update
        _update = self.update

    def upd_param(self, key, one_ft, block):
        self.key, self.one_ft, self.block = key, one_ft, block
        self.ft_pair = ([0], np.array([1]))
        if self.scaler is not None:
            ft_ind = get_ft_ind(self.key, (1, 2)[self.poly_fts], self.scaler['bf'], self.one_ft)
            if ft_ind:
                self.ft_pair = (list(ft_ind.keys()), np.array(list(ft_ind.values())))

    # NOTE: update p_{.,v}, return changes
    def update(self, G, X, A):
        X = {v: X[v] for v in A if X[v] > 0}
        if X:
            Pg = G.es['weight']
            chg = []
            for v, x in X.items():
                if self.F is not None:
                    e = np.exp(-np.sum(self.FT[v] * ((x+1)**self.ft_pair[1] - 1)))
                for u in G.neighbors(v, mode='in'):
                    k = G.get_eid(u, v)
                    if self.F is not None:
                        # Policy 1: change the relevant polynomial features
                        G.es[k]['weight'] = 1 / (1 + (1 / Pg[k] - 1) * e)
                    else:
                        # Policy 2: directly change probabilities
                        G.es[k]['weight'] = 0 if self.block else Pg[k] * (1 - 0.1 * x)
                    chg.append(G.es[k]['weight'] - Pg[k])
            if chg:
                vprint(f'Max/Min changes: {min(chg), max(chg)}', level=2)
        return G

    def xblock(self, pobj=1e-6, max_wks=0, bs=1000):
        P = self.G.es['weight']
        Xe = {}
        if self.FT is not None:
            with BoundedPool(max_wks) as pool:
                def _update_res(fut):
                    Xe.update(fut.result())
                for i in range(0, len(self.G.es.indices), bs):
                    Ib = self.G.es.indices[i:i+bs]
                    Vb = [self.G.es[k].target for k in Ib]
                    pool.submit(_search_pb, Ib, Vb, P[i:i+bs], self.FT[Vb], self.ft_pair[1], pobj).add_done_callback(_update_res)
        return Xe

if __name__ == '__main__':
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire(ProbGraph, serialize=lambda results: None)
