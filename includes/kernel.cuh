#ifndef __KERNEL__
#define __KERNEL__

#include "config.h"
#include "utils.h"

#define CalculateSumOfSquared(n, d, d_points, d_sum_of_sqr)                             \
    __CalculateSumOfSquared<<<block_16x32_distribute(n)>>>                              \
    (n, d, d_points, d_sum_of_sqr)
__global__ void __CalculateSumOfSquared(
    const int n, const int d, const float *points, float *sum_of_sqr
);

#define ComputeActualDistances(n, d_distances, d_sum_of_sqr, k)                         \
    __ComputeActualDistances<<<linearly_distribute(n * k)>>>                            \
    (n, k, d_sum_of_sqr, d_distances)
__global__ void __ComputeActualDistances(
    const int n, const int k, const float *sum_of_sqr, float *res_distances
);

#define GetDistances(j_id, i_size, j_size, d_inner_prod, d_sum_of_sqr, d_dist_block)    \
        __GetDistances<<<linearly_distribute(i_size * j_size)>>>                        \
        (j_id, i_size, j_size, d_inner_prod, d_sum_of_sqr, d_dist_block)
__global__ void __GetDistances(
    const int j_id, const int i_size, const int j_size,
    const float *inner_prod, const float *sum_of_sqr,
    float *dist
);

#define PushHeap(k, d_heap_dist, d_heap_ind, i_id, j_id, i_size, j_size, d_dist_block)  \
    __PushHeap<<<block_16x32_distribute(i_size)>>>                                      \
    (k, d_heap_dist, d_heap_ind, i_id, j_id, i_size, j_size, d_dist_block)
__global__ void __PushHeap(
    const int k,
    float *heap_dist, int *heap_ind,
    const int i_id, const int j_id,
    const int i_size, const int j_size,
    const float *dist
);

#define AssignInfinity(n, k, d_heap_dist)                                               \
    __AssignInfinity<<<linearly_distribute(n * k)>>>(n * k, d_heap_dist)
__global__ void __AssignInfinity(const int size, float *a);

#endif // __KERNEL__