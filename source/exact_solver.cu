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
    ComputeActualDistances(n, d_heap_dist, sum_of_sqr, k);
    cudaMemcpy(res_distances, d_heap_dist, n * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_indices, d_heap_ind, n * k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_heap_dist);
    cudaFree(d_heap_ind);
}

long long cu_timer;
long long get_timer;

void KnnSolver::__Solve() {
    __PreProcessing();

    const int n_blocks = intceildiv(n, BLOCK_SIZE);
    for (int i = 0; i < n_blocks; ++i) {
        const int i_size = min(BLOCK_SIZE, n - i * BLOCK_SIZE);
        const float *i_block = d_points + i * BLOCK_SIZE * d;

        cu_timer = get_timer = 0;
        long long calc_timer = 0, heap_timer = 0;

        for (int j = 0; j < n_blocks; ++j) {
            const int j_size = min(BLOCK_SIZE, n - j * BLOCK_SIZE);
            const float *j_block = d_points + j * BLOCK_SIZE * d;
            
            auto start = chrono::high_resolution_clock::now();
            __Calc(i_size, i_block, j, j_size, j_block);
            auto stop = chrono::high_resolution_clock::now();
            calc_timer += chrono::duration_cast<chrono::nanoseconds>(stop - start).count();

            start = chrono::high_resolution_clock::now();
            __Push_Heap(i, j, i_size, j_size);
            stop = chrono::high_resolution_clock::now();
            heap_timer += chrono::duration_cast<chrono::nanoseconds>(stop - start).count();
        }

        cout << heap_timer << ' ' << calc_timer << ' ' << cu_timer << ' ' << get_timer << endl;
        // cout << calc_timer << endl;
    }
    
    __PostProcessing();
}

void KnnSolver::__PreProcessing() {
    cublasCreate(&handle);

    cudaMalloc(&inner_prod, sqr(BLOCK_SIZE) * sizeof(float));

    cudaMalloc(&d_dist, sqr(BLOCK_SIZE) * sizeof(float));

    cudaMalloc(&d_heap_dist, n * k * sizeof(float));
    AssignInfinity<<<linearly_distribute(n * k)>>>(d_heap_dist);

    cudaMalloc(&d_heap_ind, n * k * sizeof(int));
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
    auto start = chrono::high_resolution_clock::now();
    cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        j_size, i_size, d,
        &alpha, j_block, d, i_block, d,
        &beta, inner_prod, j_size
    );
    auto stop = chrono::high_resolution_clock::now();
    cu_timer = chrono::duration_cast<chrono::nanoseconds>(stop - start).count();

    start = chrono::high_resolution_clock::now();
    GetDistInd(
        d_dist, inner_prod, i_size, j, j_size, sum_of_sqr
    );
    stop = chrono::high_resolution_clock::now();
    get_timer = chrono::duration_cast<chrono::nanoseconds>(stop - start).count();
}

void KnnSolver::__Push_Heap(const int i, const int j, const int i_size, const int j_size) {
    __DownHeap<<<block_16x32_distribute(i_size)>>>(
        k, d_heap_dist, d_heap_ind, i, j, i_size, j_size, d_dist
    );
}