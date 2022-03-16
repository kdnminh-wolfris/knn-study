#include <fstream>
#include <algorithm>
#include <math.h>
#include <thread>
#include <cblas.h>

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
    if (results.empty())
        fo << "This instance has not been solved!";
    else {
        for (int i = 0; i < n; ++i) {
            for (int neighbour_id: results[i])
                fo << neighbour_id << ' ';
            fo << '\n';
        }
    }
    fo.close();
}

void KnnModel::PreProcessing() {
    results.resize(n, vector<int>(k, -1));
    PreCalculationOfDistance();
}

void KnnModel::Solve() {
    PreProcessing();

    // ofstream fo("debug.out");

    pair<double, int>** dist_from_to = new pair<double, int>*[n];
    for (int i = 0; i < n; ++i)
        dist_from_to[i] = new pair<double, int>[k];

    // cout << "Done allocating memory for heaps" << endl;

    const int n_blocks = (n + block_size - 1) / block_size;
    vector<double*> blocks(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        blocks[i] = new double[min(block_size, n - i * block_size) * d];
        for (int j = 0, lim = min(block_size, n - i * block_size); j < lim; ++j)
            for (int k = 0; k < d; ++k)
                blocks[i][j * d + k] = points[(i * block_size + j) * d + k];
    }

    // cout << "Done creating blocks" << endl;

    double* foo = new double[block_size * block_size];
    for (int i = 0; i < n_blocks; ++i) {
        // cout << "i = " << i << ' '; cout.flush();
        // auto start = chrono::high_resolution_clock::now();

        for (int j = i; j < n_blocks; ++j) {
            int i_size(i < n_blocks - 1 ? block_size : n - (n_blocks - 1) * block_size);
            int j_size(j < n_blocks - 1 ? block_size : n - (n_blocks - 1) * block_size);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, i_size, j_size, d, 1, blocks[i], d, blocks[j], d, 0, foo, j_size);
            
            for (int ii = 0; ii < i_size; ++ii)
                for (int jj = (i < j ? 0 : ii + 1); jj < j_size; ++jj) {
                    int i_index = i * block_size + ii;
                    int j_index = j * block_size + jj;
                    double dist = sum_of_squared[i_index] + sum_of_squared[j_index] - 2 * foo[ii * j_size + jj];

                    // insert heap[i]
                    if (j_index <= k || dist < dist_from_to[i_index][0].first)
                        push_heap(dist_from_to[i_index], dist_from_to[i_index] + min(j_index - 1, k), k, {dist, j_index});

                    // insert heap[j]
                    if (i_index < k || dist < dist_from_to[j_index][0].first)
                        push_heap(dist_from_to[j_index], dist_from_to[j_index] + min(i_index, k), k, {dist, i_index});
                }
        }
        
        // auto stop = chrono::high_resolution_clock::now();
        // auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        // fo << duration.count() / 1000000 << '.' << duration.count() % 1000000 << 's' << endl;
    }
    delete[] foo;

    // cout << "Done solving" << endl;

    for (int i = 0; i < n; ++i) {
        sort_heap(dist_from_to[i], dist_from_to[i] + k);
        for (int j = 0; j < k; ++j)
            results[i][j] = dist_from_to[i][j].second;
    }

    // cout << "Done getting results" << endl;

    for (int i = 0; i < n; ++i)
        delete[] dist_from_to[i];
    delete[] dist_from_to;

    for (double* blk: blocks)
        delete[] blk;

    // cout << "Done unallocating memory" << endl;

    // fo.close();
}

void KnnModel::PreCalculationOfDistance() {
    sum_of_squared.resize(n, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j)
            sum_of_squared[i] += points[i * d + j] * points[i * d + j];
}

void KnnModel::Clean() {
    results.clear();
    sum_of_squared.clear();
}

KnnModel::~KnnModel() {
    delete[] points;
}

template<typename T>
void push_heap(T* it_begin, T* it_end) {
    unsigned short int cur((--it_end) - it_begin);
    while (it_end != it_begin) {
        unsigned short int par(cur); --par; par >>= 1;
        T* it_par(it_end - cur + par);
        if (*it_par < *it_end) {
            swap(*it_par, *it_end);
            cur = par;
            it_end = it_par;
        }
        else break;
    }
}

template<typename T>
void pop_heap(T* it_begin, T* it_end) {
    unsigned short int cur(0);
    swap(*it_begin, *(--it_end));
    do {
        unsigned short int left((cur << 1) | 1);
        unsigned short int right(cur); ++right; right <<= 1;
        T* it_left(it_begin - cur + left);
        T* it_right(it_begin - cur + right);
        if (it_left >= it_end)
            break;
        if (it_right < it_end && *it_left < *it_right) {
            it_left = it_right;
            left = right;
        }
        if (*it_begin < *it_left) {
            swap(*it_begin, *it_left);
            it_begin = it_left;
            cur = left;
        }
        else break;
    } while (true);
}

template<typename T>
void sort_heap(T* it_begin, T* it_end) {
    for (; it_end != it_begin; --it_end)
        pop_heap(it_begin, it_end);
}

template<typename T>
void push_heap(T* it_begin, T* it_end, int size_lim, T val) {
    if (it_end - it_begin == size_lim && val < *it_begin)
        pop_heap(it_begin, it_end);
    else ++it_end;
    *(it_end - 1) = val;

    unsigned short int cur((--it_end) - it_begin);
    while (it_end != it_begin) {
        unsigned short int par(cur); --par; par >>= 1;
        T* it_par(it_end - cur + par);
        if (*it_par < *it_end) {
            swap(*it_par, *it_end);
            cur = par;
            it_end = it_par;
        }
        else break;
    }
}