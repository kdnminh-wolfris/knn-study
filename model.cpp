#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <thread>
#include <cblas.h>
#include <time.h>
#include <stdlib.h>

#pragma GCC target("avx2")
#include <immintrin.h>

#include "model.h"

KnnModel::KnnModel() {
    // intentionally left blank
}

bool KnnModel::ReadData(string path) {
    // Simple case first
    ifstream fi(path);
    fi >> n >> d >> k;
    // n = 1000;

    points = new double[n * d];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j)
            fi >> points[i * d + j];
    fi.close();
    return true;
}

void KnnModel::Output(string path) {
    ofstream fo(path);
    if (!results)
        fo << "This instance has not been solved!";
    else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j)
                fo << results[i][j] << ' ';
            fo << '\n';
        }
    }
    fo.close();
}

void KnnModel::PreProcessing() {
    Clean();
    results = new int*[n];
    for (int i = 0; i < n; ++i)
        results[i] = new int[k];
    PreCalculationOfDistance();
}

void KnnModel::Solve() {
    PreProcessing();

    pair<double, int>** dist_from_to = new pair<double, int>*[n];
    for (int i = 0; i < n; ++i)
        dist_from_to[i] = new pair<double, int>[k];

    GetResults(dist_from_to);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j)
            results[i][j] = dist_from_to[i][j].second;
    }

    for (int i = 0; i < n; ++i)
        delete[] dist_from_to[i];
    delete[] dist_from_to;
}

long long time_cnt = 0; // for debugging
double sum = 0; // for debugging

inline void KnnModel::GetResults(pair<double, int>** dist_from_to) {
    const int n_blocks = (n + block_size - 1) / block_size;
    const int last_block_size = n - (n_blocks - 1) * block_size;

    // Allocate memory
    double** blocks = new double*[n_blocks];
    for (int i = 0; i < n_blocks - 1; ++i) {
        blocks[i] = new double[block_size * d];
        const int i_base = i * block_size;
        for (int j = 0; j < block_size; ++j)
            for (int k = 0; k < d; ++k)
                blocks[i][j * d + k] = points[(i_base + j) * d + k];
    }
    {
        const int i = n_blocks - 1;
        const int i_base = i * block_size;
        blocks[i] = new double[last_block_size * d];
        for (int j = 0; j < last_block_size; ++j)
            for (int k = 0; k < d; ++k)
                blocks[i][j * d + k] = points[(i_base + j) * d + k];
    }

    double* foo = new double[block_size * block_size];

    // Get results for each block i
    srand(time(NULL)); // initialise random seed
    for (int i = 0; i < n_blocks; ++i)
        for (int j = 0; j < n_blocks; ++j)
            CalcSortMerge(
                blocks[i], i, min(block_size, n - i * block_size),
                blocks[j], j, min(block_size, n - j * block_size),
                foo, dist_from_to
            );

    // Deallocate memory
    delete[] foo;

    for (int i = 0; i < n_blocks; ++i)
        delete[] blocks[i];
    delete[] blocks;

    this->timecnt = time_cnt;
    // cout << sum << '\n';
}

void KnnModel::CalcSortMerge(
    const double* i_block, const int i, const int i_size,
    const double* j_block, const int j, const int j_size,
    double* sum_of_products, pair<double, int>** dist_from_to
) {
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        i_size, j_size, d, 1, i_block, d, j_block, d, 0, sum_of_products, j_size
    );

    pair<double, int>* foo = new pair<double, int>[j_size];
    pair<double, int>* bar = new pair<double, int>[k];
    for (int ii = 0; ii < i_size; ++ii) {
        const int i_index = i * block_size + ii;

        for (int jj = 0; jj < j_size; ++jj)
            foo[jj] = {
                sum_of_squared[j * block_size + jj]
                - 2 * sum_of_products[ii * j_size + jj],
                j * block_size + jj
            };
        
        sort(foo, foo + j_size, k + (i == j));
        if (j == 0)
            for (int x = (i == j); x < k + (i == j); ++x)
                dist_from_to[i * block_size + ii][x - (i == j)] = foo[x];
        else {
            for (int x = 0, y = 0, z = (i == j); x < k; ++x)
                if (z >= j_size || (y < k && dist_from_to[i_index][y] < foo[z]))
                    bar[x] = dist_from_to[i_index][y++];
                else
                    bar[x] = foo[z++];
            for (int x = 0; x < k; ++x)
                dist_from_to[i_index][x] = bar[x];
        }
    }
    delete[] foo;
    delete[] bar;
}

void KnnModel::PreCalculationOfDistance() {
    sum_of_squared = new double[n];
    for (int i = 0; i < n; ++i)
        sum_of_squared[i] = 0;
    for (int i = 0, lim(n * d); i < lim; ++i)
        sum_of_squared[i / d] += points[i] * points[i];
}

void KnnModel::Clean() {
    if (results) {
        for (int i = 0; i < n; ++i)
            if (results[i]) delete[] results[i];
        delete[] results;
        results = nullptr;
    }
    if (sum_of_squared) {
        delete[] sum_of_squared;
        sum_of_squared = nullptr;
    }
}

KnnModel::~KnnModel() {
    delete[] points;
    Clean();
}

template<typename T>
T* choose_pivot(T* first, T* last) {
    return first + rand() % (last - first);
}

template<typename T>
T* partition(T* first, T* last) {
    swap(*choose_pivot(first, last), *(last - 1));
    T* pivot = first;
    for (; first < last - 1; ++first)
        if (*first <= *(last - 1)) {
            swap(*first, *pivot);
            ++pivot;
        }
    swap(*pivot, *(last - 1));
    return pivot;
}

template<typename T>
void sort(T* first, T* last, int k) {
    if (first >= last) return;
    if (k == 0) return;
    if (first == last - 1) return;
    if (first == last - 2) {
        if (*first > *(last - 1))
            swap(*first, *(last - 1));
        return;
    }
    
    T* pivot = partition(first, last);
    sort(first, pivot, min(k, int(pivot - first)));
    sort(pivot + 1, last, max(0, k - int(pivot - first) - 1));
}