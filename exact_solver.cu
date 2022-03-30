#include "exact_solver.cuh"

#include <iostream> // for debugging, may delete later
#include <stdio.h> // for debugging, may delete later

void KnnSolver::Solve() {
    PreProcessing();
    ResultInit();
    __Solve();
    CleanOnDevice();
}

#define MAX_THREADS 512

void KnnSolver::PreProcessing() {
    cudaMalloc(&d_points, n * d * sizeof(float));
    cudaMemcpy(d_points, points, n * d * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&sum_of_sqr, n * sizeof(float));
    cudaMemset(sum_of_sqr, 0, n * sizeof(float));

    CalculateSumOfSquared<<<(n * d + MAX_THREADS - 1) / MAX_THREADS, MAX_THREADS>>>
        (n, d, d_points, sum_of_sqr);
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
}

#define BLOCK_SIZE 1000

void KnnSolver::__Solve() {
    const int n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    float* foo;
    cudaMalloc(&foo, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int i = 0; i < n_blocks; ++i)
        for (int j = 0; j < n_blocks; ++j) {
            
        }

    cudaFree(foo);
    cublasDestroy(handle);
}

void KnnSolver::CalcSortMerge(
    const int i, const float* i_block, const int i_size,
    const int j, const float* j_block, const int j_size,
    float* inner_prod, cublasHandle_t handle,
    pair<float, int>* foo, pair<float, int>* d_foo
) {
    // CALCulate the inner products of each pair of points
    const float alpha = 1;
    const float beta = 0;

    cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        j_size, i_size, d,
        &alpha, j_block, d, i_block, d,
        &beta, inner_prod, j_size
    );
    
    // TODO
    // SORT neighbours of each point by their distance
    for (int ii = 0; ii < i_size; ++ii) {
        const int i_index = i * BLOCK_SIZE + ii;

        for (int jj = 0; jj < j_size; ++jj)
            foo[jj] = {
                sum_of_sqr[j * BLOCK_SIZE + jj]
                - 2 * inner_prod[ii * j_size + jj],
                j * BLOCK_SIZE + jj
            };

        cudaMemcpy(d_foo, foo, j_size * sizeof(pair<float, int>), cudaMemcpyHostToDevice);
        
    }

    // MERGE current k nearest neighbours of each point with the neighbours just
    // calculated and sorted, and keep the k nearest in the result arrays
}