#include "kernel.cuh"

#ifdef __DEBUG__
#include <stdio.h>
#endif

__global__ void __CalculateSumOfSquared(
    const int n, const int d, const float *points, float *sum_of_sqr
) {
    const int point_id = blockIdx.x * blockDim.x / WARP_SIZE + threadIdx.x / WARP_SIZE;
    if (point_id >= n) return;

    const int lane_id = threadIdx.x % WARP_SIZE;

    float sum = 0;
    for (int i = lane_id; i < d; i += WARP_SIZE)
        sum += sqr(points[point_id * d + i]);
    for (int offset = 1; offset < 32; offset <<= 1)
        sum += __shfl_down_sync(FULL_MASK, sum, offset);

    if (lane_id == 0) sum_of_sqr[point_id] = sum;
}

__global__ void __ComputeActualDistances(
    const int n, const int k, const float *sum_of_sqr, float *res_distances
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id / k >= n) return;

    __shared__ float sum[blockDim.x];
    const int relative_point_id = id / k - blockIdx.x * blockDim.x / k;
    if (threadIdx.x == 0 || id % k == 0)
        sum[relative_point_id] = sum_of_sqr[id / k];
    __syncthreads();

    res_distances[id] += sum[relative_point_id];
}

__global__ void __GetDistInd(
    float *dist, const float *inner_prod,
    const int i_size, const int j, const int j_size,
    const float *sum_of_sqr
) {
    // const int ii = blockIdx.x * blockDim.x / WARP_SIZE + threadIdx.x / WARP_SIZE;
    // const int lane_id = threadIdx.x % WARP_SIZE;
    // __shared__ float sum[128 * WARP_SIZE];

    // int k = 0;
    // int j_ext = intceildiv(j_size, 512) * 512;
    // for (int jj = lane_id; jj < j_ext; jj += WARP_SIZE) {
    //     if (k % 128 == 0) {
    //         __syncthreads();
    //         for (int i = threadIdx.x; i < min(j_size - k * 32, 128 * 32); i += blockDim.x) {
    //             sum[i] = sum_of_sqr[j * BLOCK_SIZE + i + k * 32];
    //         }
    //         __syncthreads();
    //     }
    //     ++k;
    //     if (ii < i_size && jj < j_size)
    //         dist[ii * j_size + jj] = sum[jj % (128 * 32)] - 2 * inner_prod[ii * j_size + jj];
    // }

    // #define ROW_SIZE 32
    // const int row_id =
    //     (blockIdx.x / intceildiv(j_size, ROW_SIZE)) * blockDim.x / ROW_SIZE
    //     + threadIdx.x / ROW_SIZE;
    // const int col_id =
    //     (blockIdx.x % intceildiv(j_size, ROW_SIZE)) * ROW_SIZE
    //     + threadIdx.x % ROW_SIZE;
    // if (row_id >= i_size || col_id >= j_size) return;
    
    // __shared__ float sum[ROW_SIZE];
    // if (threadIdx.x / ROW_SIZE == 0)
    //     sum[threadIdx.x % ROW_SIZE] = sum_of_sqr[j * BLOCK_SIZE + col_id];
    // __syncthreads();

    // const int ij = row_id * j_size + col_id;
    // dist[ij] = sum[threadIdx.x % ROW_SIZE] - 2 * inner_prod[ij];

    const int ij = blockIdx.x * MAX_THREADS + threadIdx.x;
    if (ij >= i_size * j_size) return;

    const int jj = j * BLOCK_SIZE + ij % j_size;
    dist[ij] = sum_of_sqr[jj] - 2 * inner_prod[ij];
}

__global__ void __DownHeap(
    const int k, float *heap_dist, int *heap_ind,
    const int block_i, const int block_j,
    const int i_size, const int j_size,
    const float *dist
) {
    const int data_id = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    const int point_id = block_i * BLOCK_SIZE + data_id;
    const int lane_id = threadIdx.x % 32;
    if (data_id >= i_size) return;
    
    heap_dist += point_id * k;
    heap_ind += point_id * k;

    for (int j = 0; j < j_size; ++j) {
        if (point_id == block_j * BLOCK_SIZE + j) continue;

        const int ptr = data_id * j_size + j;
        const float cur_dist = dist[ptr];
        if (heap_dist[0] <= dist[ptr]) continue;

        int heap_par = 0;
        int heap_ptr = 1;
        while (true) {
            // printf("%d %d %d %d %d\n", block_i, block_j, data_id, lane_id, heap_ptr);
            if (heap_ptr >= k) break;

            float max_dist = heap_ptr + lane_id >= k ? -INFINITY : heap_dist[heap_ptr + lane_id];
            int max_lane = lane_id;
            for (int offset = 1; offset < 32; offset *= 2) {
                float next_dist = __shfl_down_sync(FULL_MASK, max_dist, offset);
                int next_lane = __shfl_down_sync(FULL_MASK, max_lane, offset);
                if (max_dist < next_dist) {
                    max_dist = next_dist;
                    max_lane = next_lane;
                }
            }

            max_dist = __shfl_sync(FULL_MASK, max_dist, 0);
            if (max_dist <= cur_dist) break;

            if (lane_id == 0) {
                heap_dist[heap_par] = max_dist;
                heap_ind[heap_par] = heap_ind[heap_ptr + max_lane];
                heap_par = heap_ptr + max_lane;
                heap_ptr = heap_par * 32 + 1;
            }
            heap_par = __shfl_sync(FULL_MASK, heap_par, 0);
            heap_ptr = __shfl_sync(FULL_MASK, heap_ptr, 0);
        } // end while

        if (lane_id == 0) {
            heap_dist[heap_par] = cur_dist;
            heap_ind[heap_par] = block_j * BLOCK_SIZE + j;
        }
    }
}

__global__ void AssignInfinity(float *a) {
    a[blockIdx.x * MAX_THREADS + threadIdx.x] = INFINITY;
}