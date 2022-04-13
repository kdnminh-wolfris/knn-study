#include "model.h"
#include "exact_solver.cuh"
#include <fstream>
#include <algorithm>

void KnnModel::ReadData(const string path) {
    ifstream fi(path);
    fi >> n >> d >> k;
    points = new float[n * d];
    for (int i = 0; i < n * d; ++i)
        fi >> points[i];
    fi.close();
}

void KnnSolver::WriteResults(const string path) {
    ofstream fo(path + "/indices.out");
    for (int i = 0; i < n * k; ++i)
        fo << res_indices[i] << " \n"[i % k == k - 1];
    fo.close();

    fo.open(path + "/distances.out");
    for (int i = 0; i < n; ++i) {
        sort(res_distances + i * k, res_distances + (i + 1) * k);
        for (int j = 0; j < k; ++j)
            fo << res_distances[i * k + j] << ' ';
        fo << '\n';
    }
    fo.close();
}