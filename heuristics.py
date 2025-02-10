from ic import influence, get_inf_mode
from prob_matrix import upd_graph
from utils import Timer
import settings as st

class LazyCentrality:
    def __init__(self, G):
        self.G = G
        self._cache = {}
        self._centrality_funcs = {
            "degree": lambda: zip(self.G.degree(mode="in"), self.G.degree(mode="out")),
            "betweenness": lambda: self.G.betweenness(directed=True),
            "closeness": lambda: zip(self.G.closeness(mode="in"), self.G.closeness(mode="out")),
            "eigenvector": lambda: self.G.eigenvector_centrality(directed=True),
            "pagerank": lambda: self.G.pagerank(directed=True),
        }

    def __getitem__(self, measure):
        if measure not in self._cache:
            if measure in self._centrality_funcs:
                self._cache[measure] = list(self._centrality_funcs[measure]())
            else:
                raise KeyError(f"Unknown centrality measure: {measure}")
        return self._cache[measure]

@Timer()
def heur_centrality(G, S, measure):
    LC = LazyCentrality(G)
    degree = LC['degree']
    centrality = LC[measure]
    X = {v: 0 for v in G.vs.indices if degree[v][0] > 0 and v not in S}
    V = sorted(X, key=lambda v: centrality[v], reverse=True)
    for v in V:
        L = st.N[v]
        while X[v] < st.N[v]:
            X[v] += L
            if sum(st.cost(X[v], v) for v in X) > st.B:
                X[v] -= L
                if L == 1:
                    break
                L = 1
        if X[v] == 0:
            break
    spd_m, ub_m = influence(G, S, st.R)
    ub_mode = ub_m is not None
    obj_trace = [(spd_m, ub_m)[ub_mode]]

    ss_mode = get_inf_mode()
    spd = spd_m
    if ss_mode:
        spd, ub = influence(G, S, st.R, 0)
    spd_trace = [spd]

    X = {v: X[v] for v in X if X[v] > 0}
    if X:
        Gv = upd_graph(G.copy(), X, X)
        spd_m, ub_m = influence(Gv, S, st.R)
        obj_trace.append((spd_m, ub_m)[ub_mode])

        spd = spd_m
        if ss_mode:
            spd, ub = influence(Gv, S, st.R, 0)
        spd_trace.append(spd)
    return X, spd_trace, obj_trace
