#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ic.h"

void set_seed(unsigned int seed) {
    srand(seed);
}

// independent cascade model for graph in csr (out-neighbors)
void ic(
    int vertices, int* neighbors, int* offsets, double* weights,
    int* seeds, int num_seeds,
    bool* influenced, int* queue,
    int* stat
) {
    int front = 0, rear = 0;

    memset(influenced, 0, vertices * sizeof(bool));

    for (int i = 0; i < num_seeds; i++) {
        int v = seeds[i];
        if (!influenced[v]) {
            influenced[v] = true;
            stat[v] += 1;
            queue[rear++] = v;
        }
    }

    while (front < rear) {
        int u = queue[front++];
        for (int i = offsets[u]; i < offsets[u + 1]; i++) {
            int v = neighbors[i];
            if (!influenced[v] && ((double)rand() / RAND_MAX) < weights[i]) {
                influenced[v] = true;
                stat[v] += 1;
                queue[rear++] = v;
            }
        }
    }
}

// monte-carlo ic for graph in csr (out-neighbors)
void ic_mc(
    int vertices, int* neighbors, int* offsets, double* weights,
    int* seeds, int num_seeds,
    int num_runs,
    int* stat
) {
    bool* influenced = (bool*)malloc(vertices * sizeof(bool));
    int* queue = (int*)malloc(vertices * sizeof(int));

    for (int n = 0; n < num_runs; n++) {
        ic(
            vertices, neighbors, offsets, weights,
            seeds, num_seeds,
            influenced, queue,
            stat
        );
    }

    free(influenced);
    free(queue);
}

// steady-state spread for graph in csr (in-neighbors)
void ic_sss(
    int vertices, int* neighbors, int* offsets, double* weights,
    int* seeds, int num_seeds,
    int num_runs, double tol,
    double* prob
) {
    // mark the original pointer prob
    double* prob_o = prob;
    double* prob_n = (double*)malloc(vertices * sizeof(double));

    for (int i = 0; i < num_seeds; i++) {
        prob[seeds[i]] = 1.0;
        prob_n[seeds[i]] = 1.0;
    }

    for (int r = 0; r < num_runs; r++) {
        double max_diff = 0.0;
        for (int v = 0; v < vertices; v++) {
            int is_seed = 0;
            for (int i = 0; i < num_seeds; i++) {
                if (seeds[i] == v) {
                    is_seed = 1;
                    break;
                }
            }
            if (is_seed) continue;

            double pd = 1.0;
            for (int idx = offsets[v]; idx < offsets[v + 1]; idx++) {
                int nb = neighbors[idx];
                pd *= (1.0 - weights[idx] * prob[nb]);
            }
            prob_n[v] = 1.0 - pd;

            double diff = fabs(prob_n[v] - prob[v]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }

        double* temp = prob;
        prob = prob_n;
        prob_n = temp;

        if (max_diff < tol) {
            break;
        }
    }

    // swap back original pointer
    if (prob_o != prob) {
        memcpy(prob_o, prob, vertices * sizeof(double));
        prob_n = prob;
    }

    free(prob_n);
}
