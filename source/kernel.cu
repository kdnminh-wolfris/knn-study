#include "kernel.cuh"

#ifdef __DEBUG__
#include <stdio.h>
#endif

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
    const int i, const int k,
    const int row_start, const int row_stride,
    float *res_distances, int *res_indices,
    const float *dist, const int *ind, const int n_pts
) {
    coor_init(r, c, k);
    const int rid = (i * BLOCK_SIZE + r) * k + c;
    const int bid = r * row_stride + row_start + c;

    if (r >= n_pts) return;
    
    res_distances[rid] = dist[bid];
    res_indices[rid] = ind[bid];
}

__global__ void MergeToResults(
    const int i, const int k,
    float *res_distances, int *res_indices,
    const float *dist, const int *ind,
    const int row_start, const int row_stride, const int n_pts
) {
    if (blockIdx.x * MAX_THREADS + threadIdx.x >= n_pts) return;

    const int pt = (i * BLOCK_SIZE + blockIdx.x * MAX_THREADS + threadIdx.x) * k;
    res_distances += pt;
    res_indices += pt;
    const int bp = (blockIdx.x * MAX_THREADS + threadIdx.x) * row_stride; // block point
    dist += bp + row_start;
    ind += bp + row_start;

    int p1 = 0, p2 = 0, lim2 = row_stride - row_start;
    float d1 = res_distances[0];
    float d2 = dist[0];

    while (p1 + p2 < k)
        if (p2 == lim2 || d1 <= d2) {
            if ((++p1) + p2 < k)
                d1 = res_distances[p1];
        }
        else {
            if (p1 + (++p2) < k && p2 < lim2)
                d2 = dist[p2];
        }

    d1 = res_distances[--p1];
    d2 = dist[--p2];
    for (int x = k - 1; x >= 0; --x) {

        #ifdef __DEBUG__
        printf("%d %d %d %f %d %f\n", blockIdx.x * MAX_THREADS + threadIdx.x, x, p1, d1, p2, d2);
        #endif

        if (p2 == -1 || (p1 > -1 && d1 > d2)) {
            res_distances[x] = d1;
            res_indices[x] = res_indices[p1--];
            if (p1 > -1) d1 = res_distances[p1];
        }
        else {
            res_distances[x] = d2;
            res_indices[x] = ind[p2--];
            if (p2 > -1) d2 = dist[p2];
        }
    }
}