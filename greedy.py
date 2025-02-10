from utils import ordered_set, vprint, Timer
from ic import influence, get_inf_mode
from bounded_pool import BoundedPool
from collections import defaultdict
from prob_matrix import upd_graph
from functools import partial
import settings as st

_max_wks = 0
def set_max_wks(m):
    global _max_wks
    _max_wks = m

def _op(X, A, v, op_a, op):
    if op_a:
        (A.remove, A.add)[op](v)
    else:
        X[v] += (-1, 1)[op]

# NOTE: object will be pickled (serialized), i.e., each worker gets a separate copy in its own memory space
def _influence(G, S, X, A):
    Gv = upd_graph(G, X, A)
    return influence(Gv, S, st.R, 0)

def _update_res(fut, spd_ub_m):
    for dst, src in zip(spd_ub_m, fut.result()):
        dst.update(src)

def _influence_vb(G, Vb, S, X, A, op_a):
    spd_m, ub_m = {}, {}
    for v in Vb:
        # may add more filter
        if (op_a or X[v] < st.N[v]) and G.degree(v, mode='in') > 0:
            _op(X, A, v, op_a, 1)
            Gv = upd_graph(G.copy(), X, A)
            _op(X, A, v, op_a, 0)
            spd_m[v], ub_m[v] = influence(Gv, S, st.R)
    return spd_m, ub_m

def _influence_kb(G, Kb, S, X, Xe, Ae, pobj):
    spd_m, ub_m = {}, {}
    for k in Kb:
        v, p = G.es[k].target, G.es[k]['weight']
        if v not in S and k not in Ae and p > pobj:
            if k not in Xe or Xe[k] <= X[v] or Xe[k] > st.N[v]:
                continue
            x_org, X[v] = X[v], Xe[k]
            Gv = upd_graph(G.copy(), X, X)
            X[v] = x_org
            spd_m[k], ub_m[k] = influence(Gv, S, st.R)
    return spd_m, ub_m

def greedy_base(G, S, X, A, max_iter, spd_ub_m, spd_init, alpha, pool, bs=10):
    VS = set(G.vs.indices) - set(S)
    # NOTE: currently, initial empty A indicates operation is on A
    op_a = (len(A) == 0)
    spd_m_pre, ub_m_pre = spd_ub_m
    ub_mode = ub_m_pre is not None
    res_trace = [spd_ub_m]
    obj_trace = [spd_ub_m[ub_mode]]

    ss_mode = get_inf_mode()
    spd = spd_m_pre
    if ss_mode:
        if spd_init is None:
            spd, ub = influence(G, S, st.R, 0)
        else:
            spd = spd_init
    spd_trace = [spd]
    vprint(f'Initial spread: {spd}')

    for i in range(max_iter):
        spd_m, ub_m = {}, {}
        futs = []
        Vs = list(VS - A if op_a else A)
        for i in range(0, len(Vs), bs):
            fut = pool.submit(_influence_vb, G, Vs[i:i+bs], S, X, A, op_a)
            fut.add_done_callback(partial(_update_res, spd_ub_m=(spd_m, ub_m)))
            futs.append(fut)
        pool.wait(futs)
        if not spd_m:
            break

        # NOTE: cost or delta_cost
        obj, obj_pre = (spd_m, spd_m_pre)
        if ub_mode:
            obj, obj_pre = (ub_m, ub_m_pre)
        u = min(obj, key=lambda v: (obj[v]-obj_pre)/st.cost(X[v]+(not op_a), v))
        _op(X, A, u, op_a, 1)
        if sum(st.cost(X[v], v) for v in A) > alpha*st.B:
            _op(X, A, u, op_a, 0)
            break

        spd_m_pre, ub_m_pre = spd_m[u], ub_m[u]
        res_trace.append((spd_m_pre, ub_m_pre))
        obj_trace.append((spd_m_pre, ub_m_pre)[ub_mode])

        spd = spd_m_pre
        if ss_mode:
            spd, ub = pool.submit(_influence, G, S, X, A).result()
        spd_trace.append(spd)
        vprint(f'Decision {(u, X[u])}, with spread {spd}', flush=True)

        if spd - len(S) <= 1e-6:
            break
    return spd_trace, obj_trace, res_trace

# NOTE: if cost is same, no need to compute spd_pre
@Timer()
def greedy_filling(G, S, *_):
    VS = set(G.vs.indices) - set(S)
    X = {v: 0 for v in VS}
    spd_ub_m = influence(G, S, st.R)
    with BoundedPool(_max_wks) as pool:
        spd_trace, obj_trace, res_trace = greedy_base(
            G, S, X, VS, sum(st.N.values()), spd_ub_m, None, 1, pool
        )
    X = {v: X[v] for v in VS if X[v] > 0}
    return X, spd_trace, obj_trace

#NOTE: Deprecated
def alternating_greedy_v0(G, S, alpha):
    VS = set(G.vs.indices) - set(S)
    # NOTE: initial value > 0
    X = {v: 1 for v in VS}
    spd_ub_m = influence(G, S, st.R)
    spd_trace, obj_trace = [], []
    spd_init = None
    with BoundedPool(_max_wks) as pool:
        Ap = set()
        while alpha <= 1:
            vprint(f'Alpha: {alpha}')
            A = set()
            spd_a, obj_a, res_trace = greedy_base(
                G, S, X, A, len(VS), spd_ub_m, spd_init, alpha, pool
            )
            if spd_init is None:
                spd_init = spd_a[0]
            spd_x, obj_x, _ = greedy_base(
                G, S, X, A, sum(st.N[v] for v in A), res_trace[-1], spd_a[-1], 1, pool
            )
            spd_trace.extend([spd_a, spd_x[1:]])
            obj_trace.extend([obj_a, obj_x[1:]])
            if A == Ap:
                break
            Ap = A.copy()
            alpha += 0.1
    X = {v: X[v] for v in A if X[v] > 0}
    return X, spd_trace, obj_trace

@Timer()
def two_stage_greedy(G, S, *_):
    VS = set(G.vs.indices) - set(S)
    # NOTE: initial value > 0
    X = {v: 1 for v in VS}
    spd_ub_m = influence(G, S, st.R)
    with BoundedPool(_max_wks) as pool:
        A = set()
        spd_a, obj_a, res_a = greedy_base(
            G, S, X, A, len(VS), spd_ub_m, None, 1, pool
        )
        X = {v: 0 for v in VS}
        spd_x, obj_x, _ = greedy_base(
            G, S, X, A, sum(st.N[v] for v in A), spd_ub_m, spd_a[0], 1, pool
        )
        spd_trace, obj_trace = [spd_a, spd_x], [obj_a, obj_x]
    X = {v: X[v] for v in A if X[v] > 0}
    return X, spd_trace, obj_trace

@Timer()
def alternating_greedy(G, S, alpha, delta=0.01):
    VS = set(G.vs.indices) - set(S)
    # NOTE: initial value > 0
    X = {v: 1 for v in VS}
    spd_ub_m = influence(G, S, st.R)
    with BoundedPool(_max_wks) as pool:
        A = ordered_set()
        spd_a, obj_a, res_a = greedy_base(
            G, S, X, A, len(VS), spd_ub_m, None, 1, pool
        )

        vprint('Two-stage...')
        X = {v: 0 for v in VS}
        spd_x, obj_x, _ = greedy_base(
            G, S, X, A, sum(st.N[v] for v in A), spd_ub_m, spd_a[0], 1, pool
        )
        res_ts = ({v: X[v] for v in A if X[v] > 0}, [spd_a, spd_x], [obj_a, obj_x])
        X = {v: 1 for v in VS}

        spd_trace, obj_trace = [spd_a], [obj_a]
        spd_min, Xm, km = spd_a[-1], {v: 1 for v in A}, len(A)
        Al, k, check = list(A), 0, False
        while k <= len(A):
            cs = sum(st.cost(X[v], v) for v in Al[:k+1])
            if cs > alpha*st.B:
                vprint(f'Alpha: {alpha}, k: {k}')
                alpha += (round((cs/st.B-alpha)/delta)+1)*delta
                if not check:
                    continue
                spd_x, obj_x, _ = greedy_base(
                    G, S, X, set(Al[:k]), sum(st.N[v] for v in Al[:k]), res_a[k], spd_a[k], 1, pool
                )
                if spd_x[-1] < spd_min:
                    spd_min, Xm, km = spd_x[-1], X.copy(), k
                X = {v: 1 for v in VS}
                check = False
                spd_trace.append(spd_x)
                obj_trace.append(obj_x)
            else:
                k += 1
                check = True
            if alpha > 1:
                break
    X = {v: Xm[v] for v in Al[:km] if Xm[v] > 0}
    return X, spd_trace, obj_trace, res_ts

@Timer()
def greedy_edge_block(G, S, Xt, pobj=1e-6, bs=10):
    spd_m_pre, ub_m_pre = influence(G, S, st.R)
    ub_mode = ub_m_pre is not None
    obj_trace = [(spd_m_pre, ub_m_pre)[ub_mode]]

    ss_mode = get_inf_mode()
    spd = spd_m_pre
    if ss_mode:
        spd, ub = influence(G, S, st.R, 0)
    spd_trace = [spd]
    vprint(f'Initial spread: {spd}')

    with BoundedPool(_max_wks) as pool:
        VS = set(G.vs.indices) - set(S)
        X = {v: 0 for v in VS}
        Ae = set()
        Xe, xtype = Xt
        # prepare for target edge block
        remove = [k for k, x in Xe.items() if x < 0]
        if xtype == 1:
            # prepare for node (all incoming edges) block
            Vr = set(G.es[k].target for k in remove)
            remove = [k for k in Xe if G.es[k].target in Vr]

        for k in remove:
            Xe.pop(k)
        vprint(f'Blockable edges: {len(Xe)}')

        Xv = defaultdict(int)
        VX = defaultdict(list)
        for k, x in Xe.items():
            v = G.es[k].target
            key = (v, x)
            if xtype == 1:
                key = v
                if Xv[v] < x:
                    Xv[v] = x
            VX[key].append(k)
        if xtype == 1:
            for v, ks in VX.items():
                for k in ks:
                    Xe[k] = Xv[v]

        Ks = [vx[0] for vx in VX.values()]
        vprint(f'Ks: {len(Ks)}')
        for _ in range(len(G.es)):
            spd_m, ub_m = {}, {}
            futs = []
            for i in range(0, len(Ks), bs):
                fut = pool.submit(_influence_kb, G, Ks[i:i+bs], S, X, Xe, Ae, pobj)
                fut.add_done_callback(partial(_update_res, spd_ub_m=(spd_m, ub_m)))
                futs.append(fut)
            pool.wait(futs)
            if not spd_m:
                break

            for vx in VX.values():
                for k in vx[1:]:
                    if vx[0] in spd_m:
                        spd_m[k], ub_m[k] = spd_m[vx[0]], ub_m[vx[0]]

            obj, obj_pre = (spd_m, spd_m_pre)
            if ub_mode:
                obj, obj_pre = (ub_m, ub_m_pre)
            # can integrate constraint
            km = min(obj, key=lambda k: (obj[k]-obj_pre))
            v, u = G.es[km].tuple
            x_org, X[u] = X[u], Xe[km]
            if sum(st.cost(X[v], v) for v in VS) > st.B:
                X[u] = x_org
                break
            Ae.add(km)

            spd_m_pre, ub_m_pre = spd_m[km], ub_m[km]
            obj_trace.append((spd_m_pre, ub_m_pre)[ub_mode])

            spd = spd_m_pre
            if ss_mode:
                spd, ub = pool.submit(_influence, G, S, X, VS).result()
            spd_trace.append(spd)
            vprint(f'Decision {((v, u), X[u])}, with spread {spd}', flush=True)

            if spd - len(S) <= 1e-6:
                break
        X = {v: X[v] for v in VS if X[v] > 0}
        Xe = {G.es[k].tuple: Xe[k] for k in Ae}
    return X, spd_trace, obj_trace, Xe


def _influence_im(G, Vb, S):
    spd = {}
    for v in Vb:
        if v not in S and G.degree(v, mode='out') > 0:
            spd[v], _ = influence(G, S+[v], st.R, 0)
    return spd, {}

@Timer()
def greedy_im(G, k, bs=10):
    VS = G.vs.indices
    S, spd_trace = [], []
    with BoundedPool(_max_wks) as pool:
        for _ in range(k):
            spd, futs = {}, []
            for i in range(0, len(VS), bs):
                fut = pool.submit(_influence_im, G, VS[i:i+bs], S)
                fut.add_done_callback(partial(_update_res, spd_ub_m=(spd, {})))
                futs.append(fut)
            pool.wait(futs)
            u = max(spd, key=lambda v: spd[v])
            S.append(u)
            spd_trace.append(spd[u])
            vprint(f'Select {u} with spread {spd[u]}', flush=True)
    return S, spd_trace
