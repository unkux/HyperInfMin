from .ic_x import icm, seed_rng
import numpy as np
import time
import os

_repr_mc, _repr_ss, _ub_mode = None, None, False

# graph csr representation
class Representation:
    def __init__(self, G, mode=0):
        self.mode = mode
        self.vertices = G.vcount()
        self.adjlist = G.get_adjlist(('in', 'out')[mode == 0])
        self.edge_idx = []
        neighbors, offsets = [], [0]
        for v, nb in enumerate(self.adjlist):
            edges = (
                G.es.select(_source=v, _target_in=nb) if mode == 0
                else G.es.select(_source_in=nb, _target=v)
            )
            self.edge_idx.extend(edges.indices)
            neighbors.extend(nb)
            offsets.append(len(neighbors))
        self.neighbors = np.array(neighbors, dtype=np.int32)
        self.offsets = np.array(offsets, dtype=np.int32)

    def csr(self):
        return self.mode, self.vertices, self.neighbors, self.offsets

    # NOTE: graph topology is fixed
    def update_weights(self, G):
        return np.array(G.es['weight'], dtype=np.float64)[self.edge_idx]

def init_repr(G, mode=0):
    global _repr_mc, _repr_ss
    if _repr_mc is None:
        _repr_mc = Representation(G, 0)
    if _repr_ss is None and mode == 1:
        _repr_ss = Representation(G, 1)

def gen_seed():
    pid = os.getpid()
    now = int(time.time() * 1e6)
    seed = (pid ^ now) & 0xFFFFFFFF
    return seed

def set_ub_mode(b=True):
    global _ub_mode
    _ub_mode = b

def get_inf_mode():
    return int(_repr_ss is not None)

# NOTE: mode=None: default
def influence(G, S, R, mode=None, tol=1e-6):
    init_repr(G, mode)
    seed_rng(gen_seed())
    _repr = _repr_mc
    if mode == 1 or (mode != 0 and _repr_ss is not None):
        _repr = _repr_ss
    weights = _repr.update_weights(G)
    prob = icm(*_repr.csr(), weights, np.array(S, dtype=np.int32), len(S), R, tol)
    spd, ub = prob.sum(), None
    if _ub_mode:
        ub = -np.log((1+1e-9)-prob).sum()
    return spd, ub
