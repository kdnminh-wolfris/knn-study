#include "model.h"
#include "exact_solver.cuh"

KnnModel::~KnnModel() {
    Clean();
}

void KnnModel::Clean() {
    n = d = k = 0;
    if (points) delete[] points;
}

KnnSolver::~KnnSolver() {
    Clean();
}

void KnnSolver::Clean() {
    if (res_indices) delete[] res_indices;
    if (res_distances) delete[] res_distances;

    if (d_indices) cudaFree(d_indices);
    if (d_distances) cudaFree(d_distances);
}

void KnnSolver::CleanOnDevice() {
    cudaFree(d_points);
    cudaFree(sum_of_sqr);
}