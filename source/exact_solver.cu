#include "exact_solver.cuh"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#ifdef __DEBUG__

#include <iostream>
#include <stdio.h>
#include <algorithm>

#endif

void KnnSolver::Solve() {
    PreProcessing();
    __Solve();
    PostProcessing();
    CleanOnDevice();
}

void KnnSolver::PreProcessing() {
    cudaMalloc(&d_points, n * d * sizeof(float));
    cudaMemcpy(d_points, points, n * d * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&sum_of_sqr, n * sizeof(float));
    cudaMemset(sum_of_sqr, 0, n * sizeof(float));

    CalculateSumOfSquared<<<block_cnt(n * d), MAX_THREADS>>>(n, d, d_points, sum_of_sqr);

    ResultInit();
}

__global__ void CalculateSumOfSquared(
    const int n, const int d, const float* points, float* sum_of_sqr
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int pt = i / d;
    const int di = i % d;
    
    if (pt >= n) return;

    const bool flag = (threadIdx.x == 0 || di == 0);

    __shared__ float sos[MAX_THREADS];
    if (flag) sos[threadIdx.x / d] = 0;
    __syncthreads();

    const int rpt = threadIdx.x / d; // relative point id on sos array

    const float val = points[i];
    atomicAdd(&sos[rpt], val * val);
    __syncthreads();

    if (flag) sum_of_sqr[pt] += sos[rpt];
}

void KnnSolver::ResultInit() {
    res_indices = new int[n * k];
    res_distances = new float[n * k];

    cudaMalloc(&d_indices, n * k * sizeof(int));
    cudaMalloc(&d_distances, n * k * sizeof(float));
}

void KnnSolver::PostProcessing() {
    ComputeRealDistances<<<n, k>>>(d_distances, sum_of_sqr, k);
    cudaMemcpy(res_distances, d_distances, n * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_indices, d_indices, n * k * sizeof(int), cudaMemcpyDeviceToHost);
}

__global__ void ComputeRealDistances(float* res_distances, const float* sum_of_sqr, const int k) {
    __shared__ float this_sum_of_sqr;
    if (threadIdx.x == 0) this_sum_of_sqr = sum_of_sqr[blockIdx.x];
    __syncthreads();
    res_distances[blockIdx.x * k + threadIdx.x] += this_sum_of_sqr;
}

struct StartOp {
    const int m;
    __device__ int operator()(int x) const {
        return m * x;
    };
};

#ifdef __DEBUG__
void printArray(float* arr, int sizex, int sizey) {
    const int size = sizex * sizey;
    float* tmp = new float[sizex * sizey];
    cudaMemcpy(tmp, arr, size * sizeof(float), cudaMemcpyDeviceToHost);

    cout << sizex << ' ' << sizey << endl;
    for (int i = 0; i < size; ++i)
        cout << tmp[i] << " \n"[i % sizey == sizey - 1];
    cout << endl;
    delete[] tmp;
}
#endif

void KnnSolver::__Solve() {
    //-----Initiating values and allocating memory
    const int n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cublasCreate(&handle);

    cudaMalloc(&inner_prod, sqr(BLOCK_SIZE) * sizeof(float));

    cudaMalloc(&dist, sqr(BLOCK_SIZE) * sizeof(float));
    cudaMalloc(&ind, sqr(BLOCK_SIZE) * sizeof(int));

    cudaMalloc(&tdist, sqr(BLOCK_SIZE) * sizeof(float));
    cudaMalloc(&tind, sqr(BLOCK_SIZE) * sizeof(int));

    db_dist = {dist, tdist};
    db_ind = {ind, tind};

    cudaHostAlloc(&tmp_d, k * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&tmp_i, k * sizeof(int), cudaHostAllocDefault);
    cudaMalloc(&dtmp_d, k * sizeof(float));
    cudaMalloc(&dtmp_i, k * sizeof(int));

    cub::CountingInputIterator<int> itr(0);
    //-----Done initiating values and allocating memory

    //-----Main part of solving
    for (int i = 0; i < n_blocks; ++i) {
        const int i_size = min(BLOCK_SIZE, n - i * BLOCK_SIZE);
        const float *i_block = d_points + i * BLOCK_SIZE * d;

        for (int j = 0; j < n_blocks; ++j) {
            const int j_size = min(BLOCK_SIZE, n - j * BLOCK_SIZE);
            const float *j_block = d_points + j * BLOCK_SIZE * d;
            
            // CALCulate distances of each pair of points
            cublasSgemm(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                j_size, i_size, d,
                &alpha, j_block, d, i_block, d,
                &beta, inner_prod, j_size
            );

            GetDistInd<<<block_cnt(i_size * j_size), MAX_THREADS>>>
                (db_dist.Current(), db_ind.Current(), inner_prod, i_size, j, j_size, sum_of_sqr);

            // SORT neighbours of each point by their distance
            cub::TransformInputIterator<int, StartOp, decltype(itr)> start_itr(itr, {j_size});
            
            // if (i == 0 && j == 0) {
                cub::DeviceSegmentedSort::SortPairs(
                    nullptr, aux_size,
                    db_dist, db_ind,
                    i_size * j_size, i_size,
                    start_itr, start_itr + 1
                );
                if (aux_size > pre_aux_size) {
                    if (aux) cudaFree(aux);
                    cudaMalloc(&aux, aux_size);
                    pre_aux_size = aux_size;
                }
            // }

            cub::DeviceSegmentedSort::SortPairs(
                aux, aux_size,
                db_dist, db_ind,
                i_size * j_size, i_size,
                start_itr, start_itr + 1
            );
            
            // cout << "Block " << i << ' ' << j << endl;
            // printArray(db_dist.Current(), i_size, j_size);

            // float* tttmp_d = new float[sqr(BLOCK_SIZE)];
            // cudaMemcpy(tttmp_d, dist, i_size * j_size * sizeof(float), cudaMemcpyDeviceToHost);
            // for (int ii = 0; ii < i_size; ++ii)
            //     sort(tttmp_d + ii * j_size, tttmp_d + (ii + 1) * j_size);
            // cudaMemcpy(dist, tttmp_d, i_size * j_size * sizeof(float), cudaMemcpyHostToDevice);
            // delete[] tttmp_d;

            // MERGE current k nearest neighbours of each point with the neighbours
            // which are just calculated and sorted, and keep the k nearest in the
            // result arrays
            for (int ii = 0, i_index = i * BLOCK_SIZE; ii < i_size; ++ii, ++i_index) {
                if (j == 0)
                    AssignResults<<<1, k>>>(
                        i == j, db_dist.Current() + ii * j_size, db_ind.Current() + ii * j_size,
                        (i * BLOCK_SIZE + ii) * k, d_distances, d_indices
                    );
                else {
                    // for (int x = 0, y = 0, z = (i == j); x < k; ++x)
                    //     if (z >= j_size || (y < k && res_distances[i_index * d + y] < dist[z])) {
                    //         tmp_d[x] = res_distances[i_index * d + y];
                    //         tmp_i[x] = res_indices[i_index * d + y];
                    //         ++y;
                    //     }
                    //     else {
                    //         tmp_d[x] = dist[z];
                    //         tmp_i[x] = ind[z];
                    //         ++z;
                    //     }
                    // cudaMemcpy(dtmp_d, tmp_d, k * sizeof(float), cudaMemcpyHostToDevice);
                    // cudaMemcpy(dtmp_i, tmp_i, k * sizeof(float), cudaMemcpyHostToDevice);
                    
                    // AssignResults<<<1, k>>>(0, tmp_d, tmp_i, i_index * d, d_distances, d_indices);

                    InsertToResults(
                        db_dist.Current() + ii * j_size + (i == j), db_ind.Current() + ii * j_size + (i == j),
                        k, i_index, d_distances, d_indices
                    );

                    // cout << "-----" << endl;
                }
                
                // printArray(d_distances, n, k);
            }
        } // end for j
    } // end for i
    //-----Done solving

    //-----Deallocating memory
    cublasDestroy(handle);
    cudaFree(inner_prod);
    cudaFree(dist);
    cudaFree(ind);
    cudaFree(tdist);
    cudaFree(tind);
    cudaFreeHost(tmp_d);
    cudaFreeHost(tmp_i);
    cudaFree(dtmp_d);
    cudaFree(dtmp_i);
    if (aux) cudaFree(aux);
}

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

__host__ inline void InsertToResults(
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