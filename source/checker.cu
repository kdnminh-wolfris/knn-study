#include "exact_solver.cuh"
#include <fstream>
#include <iostream>
#include <algorithm>

template<typename T>
T* faiss_data_read(string path, int n, int k) {
    ifstream fi(path);

    T* ret = new T[n * k];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j)
            fi >> ret[i * k + j];
    }
    fi.close();

    return ret;
}

float KnnSolver::SimilarityCheck(const string path, const bool print_log, const bool by_index) {
    float similarity;
    const int all = n * k;
    int matched = 0;

    if (by_index) {
        // Read answers to check
        int* ans_indices = faiss_data_read<int>(path + "/ans_indices.out", n, k);
        
        // Check outputs
        int* foo = new int[k];
        for (int i = 0; i < n; ++i) {
            // Memberwise copy results of point i
            for (int j = 0; j < k; ++j)
                foo[j] = res_indices[i * k + j];
            
            sort(foo, foo + k);
            sort(ans_indices + i * k, ans_indices + (i + 1) * k);

            for (int j1 = 0, j2 = 0; j1 < k; ++j1) {
                for (; j2 < k && foo[j1] > ans_indices[i * k + j2]; ++j2);
                if (j2 < k && foo[j1] == ans_indices[i * k + j2]) {
                    ++matched; ++j2;
                }
            }
        }
        delete[] foo;

        // Deallocate memory
        delete[] ans_indices;
    }
    else {
        // Read answers to check
        float* ans_distances = faiss_data_read<float>(path + "/ans_indices.out", n, k);
        
        // Check outputs
        const float epsilon = 1e-3;
        for (int i = 0; i < n * k; ++i)
            matched += (fabs(ans_distances[i] - res_distances[i]) <= epsilon);

        // Deallocate memory
        delete[] ans_distances;
    }
    similarity = (float)matched / all;

    // Print detailed log
    if (print_log)
        cout << "Matched pairs: " << matched << "/" << all << endl;

    // Return checking result
    return similarity;
}

float KnnSolver::TotalDistanceDifferenceCheck(const string path, const bool print_log) {
    // Read answers to check
    float* ans_distances = faiss_data_read<float>(path + "/ans_distances.out", n, k);
    
    // Check outputs
    float total_difference = 0;

    for (int i = 0; i < n * k; ++i)
        total_difference += fabs(ans_distances[i] - res_distances[i]);

    // Print detailed log

    // Deallocate memory
    delete[] ans_distances;

    // Return checking result
    return total_difference;
}