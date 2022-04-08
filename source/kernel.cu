#include "kernel.cuh"

#ifdef __DEBUG__
#include <stdio.h>
#endif

__global__ void GetDistInd(
    float *dist, int *ind, const float *inner_prod,
    const int i_size, const int j, const int j_size,
    const float *sum_of_sqr
) {
    const int point_id = blockIdx.x * MAX_THREADS / WARP_SIZE + threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (point_id >= i_size) return;

    for (int col = lane_id; col < j_size; col += WARP_SIZE) {
        const int ij = point_id * j_size + col;
        const int jj = j * BLOCK_SIZE + col;
        dist[ij] = sum_of_sqr[jj] - 2 * inner_prod[ij];
        ind[ij] = jj;
    }
}

__global__ void AssignResults(
    const int i, const int k,
    const int row_start, const int row_stride,
    float *res_distances, int *res_indices,
    const float *dist, const int *ind, const int n_pts
) {
    // coor_init(r, c, k);
    // const int rid = (i * BLOCK_SIZE + r) * k + c;
    // const int bid = r * row_stride + row_start + c;

    // if (r >= n_pts) return;
    
    // res_distances[rid] = dist[bid];
    // res_indices[rid] = ind[bid];

    const int row_id = blockIdx.x * blockDim.x / WARP_SIZE + threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (row_id >= n_pts) return;

    const int point_ptr = (i * BLOCK_SIZE + row_id) * k;
    const int row_ptr = row_id * row_stride;

    for (int col = lane_id; col < row_start + k; col += WARP_SIZE) {
        if (col < row_start) continue;
        res_distances[point_ptr + col - row_start] = dist[row_ptr + col];
        res_indices[point_ptr + col - row_start] = ind[row_ptr + col];
    }
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