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

inline void KnnModel::SolveForHeaps(pair<double, int>** dist_from_to) {
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
            PushBlockToHeap(
                blocks[i], i, block_size, blocks[j], j, block_size,
                foo, dist_from_to
            );
        const int j = n_blocks - 1;
        PushBlockToHeap(
            blocks[i], i, block_size, blocks[j], j, last_block_size,
            foo, dist_from_to
        );
    }
    {
        const int i = n_blocks - 1;
        for (int j = i; j < n_blocks - 1; ++j)
            PushBlockToHeap(
                blocks[i], i, last_block_size, blocks[j], j, block_size,
                foo, dist_from_to
            );
        PushBlockToHeap(
            blocks[i], i, last_block_size, blocks[i], i, last_block_size,
            foo, dist_from_to
        );
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
    double* sum_of_products, pair<double, int>** dist_from_to
) {
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        i_size, j_size, d, 1, i_block, d, j_block, d, 0, sum_of_products, j_size
    );

    Heap heap;
    for (int ii = 0; ii < i_size; ++ii)
        for (int jj = (i < j ? 0 : ii + 1); jj < j_size; ++jj) {
            const int i_index = i * block_size + ii;
            const int j_index = j * block_size + jj;
            const double dist =
                sum_of_squared[i_index] + sum_of_squared[j_index]
                - 2 * sum_of_products[ii * j_size + jj];
            
            // insert heap[i]
            if (j_index <= k || dist < dist_from_to[i_index][0].first)
                heap.push_heap(
                    dist_from_to[i_index], dist_from_to[i_index] + min(j_index - 1, k),
                    k, {dist, j_index}
                );

            // insert heap[j]
            if (i_index < k || dist < dist_from_to[j_index][0].first)
                heap.push_heap(
                    dist_from_to[j_index], dist_from_to[j_index] + min(i_index, k),
                    k, {dist, i_index}
                );
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

void Heap::push_heap(
    pair<double, int>* it_begin, pair<double, int>* it_end,
    int size_lim, pair<double, int> val
) {
    if (it_end - it_begin == size_lim && val < *it_begin)
        pop_heap(it_begin, it_end);
    else ++it_end;

    int cur = (--it_end) - it_begin;
    while (cur) {
        int par = (cur - 1) >> 1; // par = (cur - 1) / 2
        if (it_begin[par] < val) {
            it_begin[cur] = it_begin[par];
            cur = par;
        }
        else break;
    }
    it_begin[cur] = val;
}

void Heap::pop_heap(pair<double, int>* it_begin, pair<double, int>* it_end) {
    swap(*it_begin, *(--it_end));
    pair<double, int> val = *it_begin;
    int cur = 0, last = it_end - it_begin;
    while (true) {
        int selected = (cur << 1) | 1;
        if (selected >= last) break;
        selected += (selected + 1 < last
            && it_begin[selected] < it_begin[selected + 1]);
        if (it_begin[selected] <= val) break;
        it_begin[cur] = it_begin[selected];
        cur = selected;
    }
    it_begin[cur] = val;
}

void Heap::sort_heap(pair<double, int>* it_begin, pair<double, int>* it_end) {
    for (; it_end != it_begin; --it_end)
        pop_heap(it_begin, it_end);
}

void StrongHeap::push_heap(
    pair<double, int>* it_begin, pair<double, int>* it_end,
    int size_lim, pair<double, int> val
) {
    if (it_end - it_begin == size_lim && val < *it_begin)
        pop_heap(it_begin, it_end);
    else ++it_end;

    int cur = (--it_end) - it_begin;
    while (cur) {
        int par = cur - 1;
        if (cur & 1) par >>= 1;

        if (it_begin[par] < val) {
            it_begin[cur] = it_begin[par];
            cur = par;
        }
        else break;
    }
    it_begin[cur] = val;
}

void StrongHeap::pop_heap(pair<double, int>* it_begin, pair<double, int>* it_end) {
    swap(*it_begin, *(--it_end));
    pair<double, int> val = *it_begin;
    int cur = 0, last = it_end - it_begin;
    while (true) {
        int selected = cur + 1;
        if (selected >= last) break;

        int left = (cur << 1) | 1;
        if (!(cur & 1)) {
            if (left >= last) break;
            selected = left;
        }
        else if (left < last && it_begin[selected] < it_begin[left])
            selected = left;
        if (it_begin[selected] < val) break;;
        it_begin[cur] = it_begin[selected];
        cur = selected;
    }
    it_begin[cur] = val;
}

SimdHeap::SimdHeap() {
    int tmp[8];
    for (int i = 0; i < 8; ++i)
        tmp[i] = i;
    for (int i = 0; i < 8; ++i) {
        if (i) tmp[i - 1] = tmp[i], tmp[i] = 0;
        sort_values[i] = _mm256_set_epi32(
            tmp[0], tmp[1], tmp[2], tmp[3],
            tmp[4], tmp[5], tmp[6], tmp[7]
        );
    }
}

void SimdHeap::push_heap(
    pair<double, int>* it_begin, pair<double, int>* it_end,
    int size_lim, pair<double, int> val
) {
    if (it_end - it_begin == size_lim && val < *it_begin)
        pop_heap(it_begin, it_end--);
    *it_end = val;

    // 1. Get indices of nodes from the new node all the way up to root
    ssize_t last = it_end - it_begin;
    uint32_t tmp[8];
    tmp[0] = last;
    for (int i = 1; i < 8; ++i)
        if (tmp[0]) tmp[i] = (tmp[i - 1] - 1) >> 1;
        else tmp[i] = 0;
    const __m256i indices = _mm256_set_epi32(
        tmp[0], tmp[1], tmp[2], tmp[3],
        tmp[4], tmp[5], tmp[6], tmp[7]
    );

    // TODO: Try divide SIMD process into two parts since cannot load 8 128-bit data

    // 2. Get values in heap corresponding to indices
    // 3. Mask the parents that violates heap property, i.e. smaller than new value,
    //    indicating the parents that the new node needs to climb over
    // 4. Climb to correct position
}