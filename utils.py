from contextlib import ContextDecorator
from collections.abc import Iterable
from multiprocessing import Value
from pprint import pprint
import pandas as pd
import numpy as np
import time
import os

if True:
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)

VERBOSE = 0
def set_verbose(v):
    global VERBOSE
    VERBOSE = v

def vprint(*arg, level=0, **kwargs):
    if VERBOSE > level:
        if arg:
            if len(arg) > 1 or isinstance(arg[0], str) or 'flush' in kwargs:
                print(*arg, **kwargs)
            else:
                pprint(*arg, **kwargs, width=200)
        else:
            print(*arg, **kwargs)

class Timer(ContextDecorator):
    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.run_time = time.perf_counter() - self._start_time
        vprint(f'Elapsed time: {self.run_time:.6f} seconds')
        return False

# process-safe
class progress:
    def __init__(self, length, precent=100, timing=False):
        self.precent = precent
        self.timing = timing
        self.cnt = Value('i', 0)
        self.set_progress(length)

    def set_progress(self, length):
        self.length, self.blk = length, max(1, int(length/self.precent))
        self.cnt.value, self.blk_cnt = 0, 0
        self.t1 = time.time()

    def update(self):
        with self.cnt.get_lock():
            self.cnt.value += 1
            if self.cnt.value % self.blk == 0:
                self.blk_cnt += 1
                vprint('.', end='', flush=True)
                if self.timing:
                    t2 = time.time()
                    vprint(t2 - self.t1)
                    self.t1 = t2
            if self.cnt.value == self.length:
                self.cnt.value, self.blk_cnt = 0, 0
                vprint()

def flatten(xs):
    # NOTE: recursive
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

# NOTE: avoid exp(x) overflow when testing a pre-trained model
XMAX = 100
def clip_max(*arrs, xmax=XMAX):
    for a in arrs:
        np.clip(a, None, xmax, out=a)

def vfilter(X, fp=1):
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return X >= np.quantile(X, 1-fp)

# return [[] if m[0]==-1 else m for m in mat]
def load_mat(filename, dtypes=[float], vfilter=lambda x: x, comment='#', delimiter=','):
    if not os.path.exists(filename):
        raise SystemExit(f'{filename} not exists!')
    data, n = [], len(dtypes)
    with open(filename, 'r') as f:
        for line in f:
            if line and line[0] != comment:
                ls = [x for x in line.split(delimiter) if x.strip()]
                if dtypes:
                    item = list(map(lambda x: dtypes[x[0] % n](x[1]), enumerate(ls)))
                else:
                    item = ls
                data.append(vfilter(item))
    return data

def str2num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None

def num2str(v, f_digit=2):
    return f'{v:.{f_digit}f}' if isinstance(v, float) else f'{v}'


class ordered_set(set):
    def __init__(self, iterable=None):
        self._ordered_dict = dict.fromkeys(iterable) if iterable else dict()
        super().__init__(self._ordered_dict.keys())

    def __iter__(self):
        return iter(self._ordered_dict.keys())

    def add(self, item):
        self._ordered_dict[item] = None
        super().add(item)

    def remove(self, item):
        if item in self:
            del self._ordered_dict[item]
            super().remove(item)
