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
            d_dist_block, d_ind_block, d_inner_prod, i_size, j, j_size, d_sum_of_sqr    \
        )                                                                               \
    __GetDistInd<<<linearly_distribute(i_size * j_size)>>>                              \
    (d_dist_block, d_ind_block, d_inner_prod, i_size, j, j_size, d_sum_of_sqr)
__global__ void __GetDistInd(
    float *dist, int *ind, const float *inner_prod,
    const int i_size, const int j, const int j_size,
    const float *sum_of_sqr
);

#define AssignResults(                                                                  \
            i, i_size, j, j_size, k, d_distances, d_indices, d_dist_block, d_ind_block  \
        )                                                                               \
        __AssignResults<<<linearly_distribute(i_size * k)>>>                             \
        (i, k, i == j, j_size, d_distances, d_indices, d_dist_block, d_ind_block, i_size)
__global__ void __AssignResults(
    const int i, const int k,
    const int row_start, const int row_stride,
    float *res_distances, int *res_indices,
    const float *dist, const int *ind, const int n_pts
);

#define MergeToResults(                                                                 \
            i, i_size, j, j_size, k, d_distances, d_indices, d_dist_block, d_ind_block  \
        )                                                                               \
    __MergeToResults<<<linearly_distribute(i_size)>>>                                   \
    (i, k, d_distances, d_indices, d_dist_block, d_ind_block, i == j, j_size, i_size)
__global__ void __MergeToResults(
    const int i, const int k,
    float *res_distances, int *res_indices,
    const float *dist, const int *ind,
    const int row_start, const int row_stride, const int n_pts
);

#endif // __KERNEL__