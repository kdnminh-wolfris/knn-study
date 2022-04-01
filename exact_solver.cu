#include "exact_solver.cuh"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <iostream> // for debugging, may delete later
#include <stdio.h> // for debugging, may delete later

#define sqr(x) (x * x)

void KnnSolver::Solve() {
    PreProcessing();
    ResultInit();
    __Solve();
    CleanOnDevice();
}

#define MAX_THREADS 512
#define CNT_BLOCKS(n) ((n + MAX_THREADS - 1) / MAX_THREADS)

void KnnSolver::PreProcessing() {
    cudaMalloc(&d_points, n * d * sizeof(float));
    cudaMemcpy(d_points, points, n * d * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&sum_of_sqr, n * sizeof(float));
    cudaMemset(sum_of_sqr, 0, n * sizeof(float));

    CalculateSumOfSquared<<<CNT_BLOCKS(n * d), MAX_THREADS>>>(n, d, d_points, sum_of_sqr);
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

#define BLOCK_SIZE 1000

void KnnSolver::__Solve() {
    const int n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // cuBLAS matrix multiplication coefficients
    const float alpha = 1;
    const float beta = 0;

    float* inner_prod;
    cudaMalloc(&inner_prod, sqr(BLOCK_SIZE) * sizeof(float));

    float* dist;
    cudaMalloc(&dist, sqr(BLOCK_SIZE) * sizeof(float));
    int* ind;
    cudaMalloc(&ind, sqr(BLOCK_SIZE) * sizeof(int));

    float* tmp_d;
    cudaHostAlloc(&tmp_d, k * sizeof(float), cudaHostAllocDefault);
    int* tmp_i;
    cudaHostAlloc(&tmp_i, k * sizeof(int), cudaHostAllocDefault);
    float* dtmp_d;
    cudaMalloc(&dtmp_d, k * sizeof(float));
    int* dtmp_i;
    cudaMalloc(&dtmp_i, k * sizeof(int));

    for (int i = 0; i < n_blocks; ++i) {
        const int i_size = min(BLOCK_SIZE, n - i * BLOCK_SIZE);
        const float* i_block = points + i * BLOCK_SIZE * d;

        for (int j = 0; j < n_blocks; ++j) {
            const int j_size = min(BLOCK_SIZE, n - j * BLOCK_SIZE);
            const float* j_block = points + j * BLOCK_SIZE * d;

            cublasSgemm(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                j_size, i_size, d,
                &alpha, j_block, d, i_block, d,
                &beta, inner_prod, j_size
            );

            // CALCulate the inner products of each pair of points
            GetDistInd<<<CNT_BLOCKS(i_size * j_size), MAX_THREADS>>>
                (dist, ind, inner_prod, i_size, j, j_size, sum_of_sqr);

            for (int ii = 0, i_index = i * BLOCK_SIZE; ii < i_size; ++ii, ++i_index) {
                // TODO: Find a way to sort by key two arrays on device memory
                // SORT neighbours of each point by their distance
                thrust::device_ptr<float> pdist(dist);
                thrust::device_ptr<int> pind(ind);
                thrust::sort_by_key(pdist + ii * j_size, pdist + (ii + 1) * j_size, pind + ii * j_size);

                cout << i << ' ' << j << ' ' << ii << endl;

                // TODO: Find a way to merge on GPU
                // MERGE current k nearest neighbours of each point with the neighbours
                // which are just calculated and sorted, and keep the k nearest in the
                // result arrays
                if (j == 0)
                    AssignResults<<<1, k>>>(
                        i == j, dist + ii * j_size, ind + ii * j_size,
                        (i * BLOCK_SIZE + ii) * d, res_distances, res_indices
                    );
                else {
                    for (int x = 0, y = 0, z = (i == j); x < k; ++x)
                        if (z >= j_size || (y < k && res_distances[i_index * d + y] < dist[z])) {
                            tmp_d[x] = res_distances[i_index * d + y];
                            tmp_i[x] = res_indices[i_index * d + y];
                            ++y;
                        }
                        else {
                            tmp_d[x] = dist[z];
                            tmp_i[x] = ind[z];
                            ++z;
                        }
                    cudaMemcpy(dtmp_d, tmp_d, k * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(dtmp_i, tmp_i, k * sizeof(float), cudaMemcpyHostToDevice);
                    
                    AssignResults<<<1, k>>>(0, tmp_d, tmp_i, i_index * d, res_distances, res_indices);
                }
            }
        } // end for j
    } // end for i

    cublasDestroy(handle);
    cudaFree(inner_prod);
    cudaFree(dist);
    cudaFree(ind);
    cudaFreeHost(tmp_d);
    cudaFreeHost(tmp_i);
    cudaFree(dtmp_d);
    cudaFree(dtmp_i);
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

    // TODO: Add shared memory

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