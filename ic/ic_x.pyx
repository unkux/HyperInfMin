# cython: language_level=3str

cimport numpy as np
import numpy as np

np.import_array()

cdef extern from "ic.h":
    void set_seed(
        unsigned int seed
    )
    void ic_mc(
        int vertices, int* neighbors, int* offsets, double* weights,
        int* seeds, int num_seeds,
        int num_runs,
        int* stat
    )
    void ic_sss(
        int vertices, int* neighbors, int* offsets, double* weights,
        int* seeds, int num_seeds,
        int num_runs, double tol,
        double* prob
    )

def seed_rng(unsigned int seed):
    set_seed(seed)

def icm(
    int mode,
    int vertices,
    np.ndarray[np.int32_t] neighbors,
    np.ndarray[np.int32_t] offsets,
    np.ndarray[np.float64_t] weights,
    np.ndarray[np.int32_t] seeds,
    int num_seeds,
    int num_runs,
    double tol
):
    cdef int* neighbors_c = <int*>np.PyArray_DATA(neighbors)
    cdef int* offsets_c = <int*>np.PyArray_DATA(offsets)
    cdef double* weights_c = <double*>np.PyArray_DATA(weights)
    cdef int* seeds_c = <int*>np.PyArray_DATA(seeds)

    if mode == 0:
        stat = np.zeros(vertices, dtype=np.int32)
        ic_mc(
            vertices, neighbors_c, offsets_c, weights_c,
            seeds_c, num_seeds, num_runs,
            <int*> np.PyArray_DATA(stat)
        )
        prob = stat / num_runs
    else:
        prob = np.zeros(vertices, dtype=np.float64)
        ic_sss(
            vertices, neighbors_c, offsets_c, weights_c,
            seeds_c, num_seeds, num_runs, tol,
            <double*> np.PyArray_DATA(prob)
        )
    return prob
