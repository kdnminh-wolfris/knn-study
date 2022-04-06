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
    __PreProcessing();

    for (int i = 0; i < intceildiv(n, BLOCK_SIZE); ++i) {
        const int i_size = min(BLOCK_SIZE, n - i * BLOCK_SIZE);
        const float *i_block = d_points + i * BLOCK_SIZE * d;

        for (int j = 0; j < intceildiv(n, BLOCK_SIZE); ++j) {
            const int j_size = min(BLOCK_SIZE, n - j * BLOCK_SIZE);
            const float *j_block = d_points + j * BLOCK_SIZE * d;
            
            // CALCulate distances of each pair of points
            __Calc(i_size, i_block, j, j_size, j_block);

            // SORT neighbours of each point by their distance
            __Sort(i_size, j_size);

            // MERGE current k nearest neighbours of each point with the neighbours
            // which are just calculated and sorted, and keep the k nearest in the
            // result arrays
            __Merge(i, i_size, j, j_size);
        }
    }
    
    __PostProcessing();
}

void KnnSolver::__PreProcessing() {
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
}

void KnnSolver::__PostProcessing() {
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

    GetDistInd<<<block_cnt(i_size * j_size), MAX_THREADS>>>
        (db_dist.Current(), db_ind.Current(), inner_prod, i_size, j, j_size, sum_of_sqr);
}

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
    for (int ii = 0, i_index = i * BLOCK_SIZE; ii < i_size; ++ii, ++i_index)
        if (j == 0)
            AssignResults<<<1, k>>>(
                i == j, db_dist.Current() + ii * j_size, db_ind.Current() + ii * j_size,
                (i * BLOCK_SIZE + ii) * k, d_distances, d_indices
            );
        else
            InsertToResults(
                db_dist.Current() + ii * j_size + (i == j), db_ind.Current() + ii * j_size + (i == j),
                k, i_index, d_distances, d_indices
            );
}