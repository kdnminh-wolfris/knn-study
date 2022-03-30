#include "exact_solver.cuh"
#include <fstream>
#include <iostream>
#include <algorithm>

template<typename T>
T* faiss_data_read(string path, int n, int k) {
    ifstream fi(path);
    string foo; getline(fi, foo); // ignore first line, may delete later

    T* ret = new T[n * k];

    for (int i = 0; i < n; ++i) {
        T foo; fi >> foo; // ignore first number of each line, may delete later
        for (int j = 0; j < k; ++j)
            fi >> ret[i * k + j];
    }
    fi.close();

    return ret;
}

float KnnSolver::SimilarityCheck(string path, bool print_log) {
    // Read Faiss' output to check
    int* faiss_indices = faiss_data_read<int>(path + "faiss_indices.out", n, k);
    
    // Check outputs
    float similarity;
    int all = n * k;
    int matched = 0;

    int* foo = new int[k];
    for (int i = 0; i < n; ++i) {
        // Memberwise copy results of point i
        for (int j = 0; j < k; ++j)
            foo[j] = res_indices[i * k + j];
        
        sort(foo, foo + k);
        sort(faiss_indices + i * n, faiss_indices + i * n + k);

        for (int j1 = 0, j2 = 0; j1 < k; ++j1) {
            for (; j2 < k && foo[j1] > faiss_indices[i * k + j2]; ++j2);
            if (j2 < k && foo[j1] == faiss_indices[i * k + j2]) {
                ++matched; ++j2;
            }
        }
    }
    delete[] foo;

    similarity = (float)matched / all;

    // Print detailed log
    if (print_log)
        cout << "Matched pairs: " << matched << "/" << all << endl;

    // Deallocate memory
    delete[] faiss_indices;

    // Return checking result
    return similarity;
}