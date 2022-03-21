#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <thread>
#include <cblas.h>

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

    SolveForHeaps(dist_from_to);

    for (int i = 0; i < n; ++i) {
        sort_heap(dist_from_to[i], dist_from_to[i] + k);
        for (int j = 0; j < k; ++j)
            results[i][j] = dist_from_to[i][j].second;
    }

    for (int i = 0; i < n; ++i)
        delete[] dist_from_to[i];
    delete[] dist_from_to;
}

long long time_cnt = 0;
double sum = 0;

inline void KnnModel::SolveForHeaps(pair<double, int>** heap) {
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

    // Solve for heaps in block i and block j
    for (int i = 0; i < n_blocks - 1; ++i) {
        for (int j = i; j < n_blocks - 1; ++j)
            PushBlockToHeap(blocks[i], i, block_size, blocks[j], j, block_size, foo, heap);
        const int j = n_blocks - 1;
        PushBlockToHeap(blocks[i], i, block_size, blocks[j], j, last_block_size, foo, heap);        
    }
    {
        const int i = n_blocks - 1;
        for (int j = i; j < n_blocks - 1; ++j)
            PushBlockToHeap(blocks[i], i, last_block_size, blocks[j], j, block_size, foo, heap);
        PushBlockToHeap(blocks[i], i, last_block_size, blocks[i], i, last_block_size, foo, heap);
    }

    // Deallocate memory
    delete[] foo;

    for (int i = 0; i < n_blocks; ++i)
        delete[] blocks[i];
    delete[] blocks;

    this->timecnt = time_cnt;
    // cout << sum << '\n';
}


void KnnModel::PushBlockToHeap(
    const double* i_block, const int i, const int i_size,
    const double* j_block, const int j, const int j_size,
    double* sum_of_products, pair<double, int>** heap
) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, i_size, j_size, d, 1, i_block, d, j_block, d, 0, sum_of_products, j_size);
    
    for (int ii = 0; ii < i_size; ++ii)
        for (int jj = (i < j ? 0 : ii + 1); jj < j_size; ++jj) {
            const int i_index = i * block_size + ii;
            const int j_index = j * block_size + jj;
            const double dist = sum_of_squared[i_index] + sum_of_squared[j_index] - 2 * sum_of_products[ii * j_size + jj];
            
            // insert heap[i]
            if (j_index <= k || dist < heap[i_index][0].first)
                push_heap(heap[i_index], heap[i_index] + min(j_index - 1, k), k, {dist, j_index});

            // insert heap[j]
            if (i_index < k || dist < heap[j_index][0].first)
                push_heap(heap[j_index], heap[j_index] + min(i_index, k), k, {dist, i_index});
        }
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

void doubly_push_heap(
    pair<double, int>* i_begin, pair<double, int>* i_end,
    pair<double, int>* j_begin, pair<double, int>* j_end,
    int size_lim, pair<double, int> val
) {
    if (i_end - i_begin == size_lim && val < *i_begin)
        pop_heap(i_begin, i_end);
    else ++i_end;
    *(i_end - 1) = val;

    if (j_end - j_begin == size_lim && val < *j_begin)
        pop_heap(j_begin, j_end);
    else ++j_end;
    *(j_end - 1) = val;
    
    
}

void push_heap(
    pair<double, int>* it_begin, pair<double, int>* it_end,
    int size_lim, pair<double, int> val
) {
    // auto start = chrono::high_resolution_clock::now();
    if (it_end - it_begin == size_lim && val < *it_begin)
        pop_heap(it_begin, it_end);
    else ++it_end;
    *(it_end - 1) = val;

    unsigned short int cur((--it_end) - it_begin);
    while (it_end != it_begin) {
        unsigned short int par(cur); --par; par >>= 1; // par = (cur - 1) / 2
        pair<double, int>* it_par(it_end - cur + par);
        if (*it_par < *it_end) {
            swap(*it_par, *it_end);
            cur = par;
            it_end = it_par;
        }
        else break;
    }
    // auto stop = chrono::high_resolution_clock::now();
    // time_cnt += (long long)(chrono::duration_cast<chrono::nanoseconds>(stop - start).count());
}

void pop_heap(pair<double, int>* it_begin, pair<double, int>* it_end) {
    swap(*it_begin, *(--it_end));
    unsigned short int cur = 0, last = it_end - it_begin;
    do {
        unsigned short int left((cur << 1) | 1); // left = cur * 2 + 1
        if (left >= last) break;

        unsigned short int right(left); ++right; // right = left + 1
        if ((right >= last || it_begin[left].first > it_begin[right].first)
            && it_begin[cur].first < it_begin[left].first) {
            swap(it_begin[cur], it_begin[left]);
            cur = left;
            continue;
        }

        swap(it_begin[cur], it_begin[right]);
        cur = right;
    } while (true);
}

void sort_heap(pair<double, int>* it_begin, pair<double, int>* it_end) {
    for (; it_end != it_begin; --it_end)
        pop_heap(it_begin, it_end);
}