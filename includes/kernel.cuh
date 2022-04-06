#ifndef __KERNEL__
#define __KERNEL__

#include "config.h"

__global__ void CalculateSumOfSquared(
    const int n, const int d, const float* points, float* sum_of_sqr
);

__global__ void GetDistInd(
    float* dist, int* ind, const float* inner_prod,
    const int i_size, const int j, const int j_size,
    const float* sum_of_sqr
);

__global__ void AssignResults(
    const int start_i, const float* dist, const int* ind,
    const int start_j, float* res_distances, int* res_indices
);

__host__ void InsertToResults(
    const float* sorted_dist, const int* sorted_ind,
    const int k, const int i, float* res_distances, int* res_indices
);

__global__ void InsertToResultWarp(
    const float *insert_dist, const int *insert_ind,
    const int start_i, float* res_distances, int* res_indices
);

__global__ void ComputeRealDistances(float* res_distances, const float* sum_of_sqr, const int k);

#endif // __KERNEL__