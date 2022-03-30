#include "model.h"
#include "exact_solver.cuh"
#include <fstream>

void KnnModel::ReadData(string path) {
    ifstream fi(path);
    fi >> n >> d >> k;
    points = new float[n * d];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j)
            fi >> points[i * d + j];
    fi.close();
}

void KnnSolver::WriteResults(string path) {
    ofstream fo(path + "indices.out");
    for (int i = 0; i < n * d; ++i)
        fo << res_indices[i] << " \n"[i % d == d - 1];
    fo.close();
    fo.open(path + "distances.out");
    for (int i = 0; i < n * d; ++i)
        fo << res_distances[i] << " \n"[i % d == d - 1];
    fo.close();
}