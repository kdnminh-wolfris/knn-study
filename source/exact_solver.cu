#include "exact_solver.cuh"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#ifdef __DEBUG__

#include <iostream>
#include <stdio.h>
#include <algorithm>

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

    CalculateSumOfSquared(n, d, d_points, sum_of_sqr);

    ResultInit();
}

void KnnSolver::ResultInit() {
    res_indices = new int[n * k];
    res_distances = new float[n * k];

    cudaMalloc(&d_indices, n * k * sizeof(int));
    cudaMalloc(&d_distances, n * k * sizeof(float));
}

void KnnSolver::PostProcessing() {
    ComputeActualDistances(n, d_distances, sum_of_sqr, k);
    cudaMemcpy(res_distances, d_distances, n * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_indices, d_indices, n * k * sizeof(int), cudaMemcpyDeviceToHost);
}

void KnnSolver::__Solve() {
    __PreProcessing();

    const int n_blocks = intceildiv(n, BLOCK_SIZE);
    for (int i = 0; i < n_blocks; ++i) {
        const int i_size = min(BLOCK_SIZE, n - i * BLOCK_SIZE);
        const float *i_block = d_points + i * BLOCK_SIZE * d;

        for (int j = 0; j < n_blocks; ++j) {
            const int j_size = min(BLOCK_SIZE, n - j * BLOCK_SIZE);
            const float *j_block = d_points + j * BLOCK_SIZE * d;

            __Calc(i_size, i_block, j, j_size, j_block);
            __Push_Heap(i, j, i_size, j_size);
        }
    }
    
    __PostProcessing();
}

void KnnSolver::__PreProcessing() {
    cublasCreate(&handle);
    cudaMalloc(&inner_prod, sqr(BLOCK_SIZE) * sizeof(float));
    cudaMalloc(&d_dist, sqr(BLOCK_SIZE) * sizeof(float));
    AssignInfinity(n, k, d_distances);
}

void KnnSolver::__PostProcessing() {
    cublasDestroy(handle);
    cudaFree(inner_prod);
    if (aux) cudaFree(aux);
    cudaFree(d_dist);
}

void KnnSolver::__Calc(
    const int i_size, const float *i_block,
    const int j, const int j_size, const float *j_block
) {
    cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        j_size, i_size, d,
        &alpha, j_block, d, i_block, d,
        &beta, inner_prod, j_size
    );

    GetDistances(j, i_size, j_size, inner_prod, sum_of_sqr, d_dist);
}

void KnnSolver::__Push_Heap(const int i, const int j, const int i_size, const int j_size) {
    PushHeap(k, d_distances, d_indices, i, j, i_size, j_size, d_dist);
}