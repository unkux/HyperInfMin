F_FT_SC = 'fmat_%s.pkl'
F_THETA = 'theta_%s.npy'
F_PMAT = 'pmat_%s.mtx'

# level of change
N = {}
# cost of change
C = {}
# budget of change
B = 0
# monte carlo runnings
R = 0

def set_inst(G, **kwargs):
    global N, C, B, R
    N, C = (kwargs.get(k, None) for k in ['N', 'C'])
    B, R, nv, cv = (kwargs.get(k, 0) for k in ['B', 'R', 'nv', 'cv'])
    block = kwargs.get('block', False)

    if N is None:
        N = {v: (nv, 1)[block] for v in G.vs.indices}
    if C is None:
        C = {v: cv for v in G.vs.indices}

# x in [0, Nv]
def cost(x, v):
    return C[v] * x
