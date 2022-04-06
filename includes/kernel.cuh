#ifndef __KERNEL__
#define __KERNEL__

#include "config.h"
#include "utils.h"

__global__ void CalculateSumOfSquared(
    const int n, const int d, const float* points, float* sum_of_sqr
);

__global__ void GetDistInd(
    float* dist, int* ind, const float* inner_prod,
    const int i_size, const int j, const int j_size,
    const float* sum_of_sqr
);

__global__ void AssignResults(
    const int i, const int k,
    const int row_start, const int row_stride,
    float *res_distances, int *res_indices,
    const float *dist, const int *ind, const int n_pts
);

__global__ void MergeToResults(
    const int i, const int k,
    float *res_distances, int *res_indices,
    const float *dist, const int *ind,
    const int row_start, const int row_stride, const int n_pts
);

__global__ void ComputeRealDistances(float* res_distances, const float* sum_of_sqr, const int k);

#endif // __KERNEL__