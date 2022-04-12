#ifndef __KERNEL__
#define __KERNEL__

#include "config.h"
#include "utils.h"

#define CalculateSumOfSquared(n, d, d_points, d_sum_of_sqr)                             \
    __CalculateSumOfSquared<<<block_16x32_distribute(n)>>>(n, d, d_points, d_sum_of_sqr)
__global__ void __CalculateSumOfSquared(
    const int n, const int d, const float *points, float *sum_of_sqr
);

#define ComputeActualDistances(n, d_distances, d_sum_of_sqr, k)                      \
    __ComputeActualDistances<<<linearly_distribute(n * k)>>>(d_distances, d_sum_of_sqr, n, k)
__global__ void __ComputeActualDistances(
    float *res_distances, const float *sum_of_sqr, const int n, const int k
);

#define GetDistInd(                                                                     \
            d_dist_block, d_inner_prod, i_size, j, j_size, d_sum_of_sqr    \
        )                                                                               \
    __GetDistInd<<<linearly_distribute(i_size * j_size)>>>                              \
    (d_dist_block, d_inner_prod, i_size, j, j_size, d_sum_of_sqr)
__global__ void __GetDistInd(
    float *dist, const float *inner_prod,
    const int i_size, const int j, const int j_size,
    const float *sum_of_sqr
);

__global__ void __DownHeap(
    const int k, float *heap_dist, int *heap_ind,
    const int block_i, const int block_j,
    const int i_size, const int j_size,
    const float *dist
);

__global__ void AssignInfinity(float *a);

#endif // __KERNEL__