#include "kernel.cuh"

#ifdef __DEBUG__
#include <stdio.h>
#endif

__global__ void __CalculateSumOfSquared(
    const int n, const int d, const float *points, float *sum_of_sqr
) {
    const int point_id = blockIdx.x * MAX_THREADS / WARP_SIZE + threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (point_id >= n) return;

    float s = 0;
    for (int i = lane_id; i < d; i += WARP_SIZE)
        s += sqr(points[point_id * d + i]);
    for (int offset = 1; offset < 32; offset <<= 1)
        s += __shfl_down_sync(FULL_MASK, s, offset);
    if (lane_id == 0) sum_of_sqr[point_id] = s;
}

__global__ void __ComputeActualDistances(
    float *res_distances, const float *sum_of_sqr, const int n, const int k
) {
    const int i = blockIdx.x * MAX_THREADS + threadIdx.x;
    if (i / k >= n) return;

    __shared__ float sqrsum[MAX_THREADS];
    const int relative_id = i / k - blockIdx.x * MAX_THREADS / k;
    if (threadIdx.x == 0 || i % k == 0)
        sqrsum[relative_id] = sum_of_sqr[i / k];
    __syncthreads();

    res_distances[i] += sqrsum[relative_id];
}

__global__ void __GetDistInd(
    float *dist, int *ind, const float *inner_prod,
    const int i_size, const int j, const int j_size,
    const float *sum_of_sqr
) {
    const int ij = blockIdx.x * MAX_THREADS + threadIdx.x;
    const int jj = j * BLOCK_SIZE + ij % j_size;
    if (ij >= i_size * j_size) return;
    dist[ij] = sum_of_sqr[jj] - 2 * inner_prod[ij];
    ind[ij] = jj;
}

__global__ void __AssignResults(
    const int i, const int k,
    const int row_start, const int row_stride,
    float *res_distances, int *res_indices,
    const float *dist, const int *ind, const int n_pts
) {
    const int ij = blockIdx.x * MAX_THREADS + threadIdx.x;
    if (ij >= n_pts * k) return;

    const int res_ij = i * BLOCK_SIZE * k + ij;
    const int matrix_ij = ij / k * row_stride + row_start + ij % k;

    res_distances[res_ij] = dist[matrix_ij];
    res_indices[res_ij] = ind[matrix_ij];
}

__global__ void __MergeToResults(
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