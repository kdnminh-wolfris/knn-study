#include "kernel.cuh"

__global__ void GetDistInd(
    float* dist, int* ind, const float* inner_prod,
    const int i_size, const int j, const int j_size,
    const float* sum_of_sqr
) {
    const int ii = (blockIdx.x * blockDim.x + threadIdx.x) / j_size;
    const int jj = (blockIdx.x * blockDim.x + threadIdx.x) % j_size;
    if (ii >= i_size) return;

    const int x = ii * j_size + jj;

    // TODO: Add shared memory for sum of squared

    dist[x] = sum_of_sqr[j * BLOCK_SIZE + jj] - 2 * inner_prod[x];
    ind[x] = j * BLOCK_SIZE + jj;
}

__global__ void AssignResults(
    const int start_i, const float* dist, const int* ind,
    const int start_j, float* res_distances, int* res_indices
) {
    res_distances[start_j + threadIdx.x] = dist[start_i + threadIdx.x];
    res_indices[start_j + threadIdx.x] = ind[start_i + threadIdx.x];
}

__host__ void InsertToResults(
    const float* sorted_dist, const int* sorted_ind,
    const int k, const int i, float* res_distances, int* res_indices
) {
    for (int x = 0; x < k; ++x)
        InsertToResultWarp<<<1, k>>>(
            sorted_dist + x, sorted_ind + x,
            i * k, res_distances, res_indices
        );
}

__global__ void InsertToResultWarp(
    const float *insert_dist, const int *insert_ind,
    const int start_i, float* res_distances, int* res_indices
) {
    const int i = threadIdx.x;

    float cur_dist = res_distances[start_i + i];
    int cur_ind = res_indices[start_i + i];

    float pre_dist = __shfl_up_sync(0xffffffff, cur_dist, 1);
    int pre_ind = __shfl_up_sync(0xffffffff, cur_ind, 1);

    if (i % WARP_SIZE == 0) {
        if (i == 0) {
            pre_dist = *insert_dist;
            pre_ind = *insert_ind;
        }
        else {
            pre_dist = res_distances[i - 1];
            pre_ind = res_indices[i - 1];
        }
    }

    __syncthreads();

    if (cur_dist > *insert_dist) {
        if (pre_dist <= *insert_dist) {
            res_distances[start_i + i] = *insert_dist;
            res_indices[start_i + i] = *insert_ind;
        }
        else {
            res_distances[start_i + i] = pre_dist;
            res_indices[start_i + i] = pre_ind;
        }
    }
}