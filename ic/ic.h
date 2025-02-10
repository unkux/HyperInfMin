#ifndef IC_H
#define IC_H

#include <stdbool.h>

void set_seed(unsigned int seed);

void ic(
    int vertices, int* neighbors, int* offsets, double* weights,
    int* seeds, int num_seeds,
    bool* influenced, int* queue,
    int* stat
);

void ic_mc(
    int vertices, int* neighbors, int* offsets, double* weights,
    int* seeds, int num_seeds,
    int num_runs,
    int* stat
);

void ic_sss(
    int vertices, int* neighbors, int* offsets, double* weights,
    int* seeds, int num_seeds,
    int num_runs, double tol,
    double* prob
);

#endif // IC_H
