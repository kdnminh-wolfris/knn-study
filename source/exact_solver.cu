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

    for (int i = 0; i < intceildiv(n, BLOCK_SIZE); ++i) {
        const int i_size = min(BLOCK_SIZE, n - i * BLOCK_SIZE);
        const float *i_block = d_points + i * BLOCK_SIZE * d;

        for (int j = 0; j < intceildiv(n, BLOCK_SIZE); ++j) {
            const int j_size = min(BLOCK_SIZE, n - j * BLOCK_SIZE);
            const float *j_block = d_points + j * BLOCK_SIZE * d;
            
            __Calc(i_size, i_block, j, j_size, j_block);
            __Sort(i_size, j_size);
            __Merge(i, i_size, j, j_size);

            // cout << '\n' << i << ' ' << j << '\n' << endl;
            // printArray(dist, i_size, j_size);
            // printArray(d_distances, n, k);
        }
    }
    
    __PostProcessing();
}

void KnnSolver::__PreProcessing() {
    cublasCreate(&handle);

    cudaMalloc(&inner_prod, sqr(BLOCK_SIZE) * sizeof(float));

    db_dist.selector = 0;
    cudaMalloc(&db_dist.d_buffers[0], sqr(BLOCK_SIZE) * sizeof(float));
    cudaMalloc(&db_dist.d_buffers[1], sqr(BLOCK_SIZE) * sizeof(float));

    db_ind.selector = 0;
    cudaMalloc(&db_ind.d_buffers[0], sqr(BLOCK_SIZE) * sizeof(int));
    cudaMalloc(&db_ind.d_buffers[1], sqr(BLOCK_SIZE) * sizeof(int));
}

void KnnSolver::__PostProcessing() {
    cublasDestroy(handle);
    cudaFree(inner_prod);
    if (aux) cudaFree(aux);
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

    GetDistInd(
        db_dist.Current(), db_ind.Current(),
        inner_prod, i_size, j, j_size, sum_of_sqr
    );
}

struct StartOp {
    const int m;
    __device__ int operator()(int x) const {
        return m * x;
    };
};

void KnnSolver::__Sort(const int i_size, const int j_size) {
    cub::TransformInputIterator<int, StartOp, decltype(itr)> start_itr(itr, {j_size});

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

    cub::DeviceSegmentedSort::SortPairs(
        aux, aux_size,
        db_dist, db_ind,
        i_size * j_size, i_size,
        start_itr, start_itr + 1
    );
}

void KnnSolver::__Merge(
    const int i, const int i_size,
    const int j, const int j_size
) {
    if (j == 0)
        AssignResults(
            i, i_size, j, j_size,
            k, d_distances, d_indices,
            db_dist.Current(), db_ind.Current()
        );
    else
        MergeToResults(
            i, i_size, j, j_size,
            k, d_distances, d_indices,
            db_dist.Current(), db_ind.Current()
        );
}