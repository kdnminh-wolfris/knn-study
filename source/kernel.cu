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

    __shared__ float sum[MAX_THREADS];
    const int relative_point_id = id / k - blockIdx.x * blockDim.x / k;
    if (threadIdx.x == 0 || id % k == 0)
        sum[relative_point_id] = sum_of_sqr[id / k];
    __syncthreads();

    res_distances[id] += sum[relative_point_id];
}

__global__ void __GetDistances(
    const int j_id, const int i_size, const int j_size,
    const float *inner_prod, const float *sum_of_sqr,
    float *dist
) {
    const int ij = blockIdx.x * blockDim.x + threadIdx.x;
    if (ij >= i_size * j_size) return;

    const int j_ptr = j_id * BLOCK_SIZE + ij % j_size;
    dist[ij] = sum_of_sqr[j_ptr] - 2 * inner_prod[ij];
}

__global__ void __PushHeap(
    const int k,
    float *heap_dist, int *heap_ind,
    const int i_id, const int j_id,
    const int i_size, const int j_size,
    const float *dist
) {
    const int data_id = blockIdx.x * blockDim.x / WARP_SIZE + threadIdx.x / WARP_SIZE;
    if (data_id >= i_size) return;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int point_id = i_id * BLOCK_SIZE + data_id;
    
    // Align pointers
    heap_dist += point_id * k;
    heap_ind += point_id * k;

    for (int j = 0; j < j_size; ++j) {
        // Ignore a point neighbouring itself
        if (point_id == j_id * BLOCK_SIZE + j) continue;

        const int ij = data_id * j_size + j;
        const float insert_dist = dist[ij];
        // Ignore if inserting distance isn't smaller than current heap top
        if (heap_dist[0] <= insert_dist) continue;

        int parent_ptr = 0;
        int children_ptr = 1;
        while (children_ptr < k) {
            float max_dist =
                children_ptr + lane_id < k ?
                heap_dist[children_ptr + lane_id] : -INFINITY;

            // Get child node with maximum distance
            int max_lane = lane_id;
            for (int offset = 1; offset < 32; offset <<= 1) {
                float next_dist = __shfl_down_sync(FULL_MASK, max_dist, offset);
                int next_lane = __shfl_down_sync(FULL_MASK, max_lane, offset);
                if (max_dist < next_dist) {
                    max_dist = next_dist;
                    max_lane = next_lane;
                }
            }

            max_dist = __shfl_sync(FULL_MASK, max_dist, 0);
            if (max_dist <= insert_dist) break;

            // Move child heap node with max distance up and update pointers
            if (lane_id == 0) {
                heap_dist[parent_ptr] = max_dist;
                heap_ind[parent_ptr] = heap_ind[children_ptr + max_lane];
                parent_ptr = children_ptr + max_lane;
                children_ptr = parent_ptr * WARP_SIZE + 1;
            }
            parent_ptr = __shfl_sync(FULL_MASK, parent_ptr, 0);
            children_ptr = __shfl_sync(FULL_MASK, children_ptr, 0);
        } // end while
        
        // Insert current distance and index into heap
        if (lane_id == 0) {
            heap_dist[parent_ptr] = insert_dist;
            heap_ind[parent_ptr] = j_id * BLOCK_SIZE + j;
        }
    } // end for
}

__global__ void __AssignInfinity(const int size, float *a) {
    const int id = blockIdx.x * MAX_THREADS + threadIdx.x;
    if (id >= size) return;
    a[id] = INFINITY;
}