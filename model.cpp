#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <thread>
#include <cblas.h>
#include <time.h>
#include <stdlib.h>
#include <iomanip>

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

    points = new float[n * d];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j)
            fi >> points[i * d + j];
    fi.close();
    return true;
}

void KnnModel::Output(string path) {
    if (!knn_indices)
        cout << "\nThis instance has not been solved!" << endl;
    else {
        ofstream fo(path + "indices.out");
        fo << fixed << setprecision(5);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j)
                fo << knn_indices[i][j] << ' ';
            fo << '\n';
        }
        fo.close();
        fo.open(path + "distances.out");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j)
                fo << knn_distances[i][j] << ' ';
            fo << '\n';
        }
        fo.close();
    }
}

template<typename T>
T** faiss_data_read(string path, int n, int k) {
    ifstream fi(path);
    string foo; getline(fi, foo); // ignore first line

    T** ret = new T*[n];
    for (int i = 0; i < n; ++i)
        ret[i] = new T[k];

    for (int i = 0; i < n; ++i) {
        T foo; fi >> foo; // ignore first number of each line
        for (int j = 0; j < k; ++j)
            fi >> ret[i][j];
    }
    fi.close();

    return ret;
}

float KnnModel::SimilarityCheck(string indices_path, bool print_log) {
    // Read output to check
    int** faiss_indices = faiss_data_read<int>(indices_path, n, k);
    
    // Check outputs
    float similarity;
    int all = n * k;
    int matched = 0;

    int* foo = new int[k];
    for (int i = 0; i < n; ++i) {
        // Memberwise copy results of point i
        for (int j = 0; j < k; ++j)
            foo[j] = knn_indices[i][j];
        
        sort(foo, foo + k);
        sort(faiss_indices[i], faiss_indices[i] + k);

        for (int j1 = 0, j2 = 0; j1 < k; ++j1) {
            for (; j2 < k && foo[j1] > faiss_indices[i][j2]; ++j2);
            if (j2 < k && foo[j1] == faiss_indices[i][j2]) {
                ++matched; ++j2;
            }
        }
    }
    delete[] foo;

    similarity = (float)matched / all;

    // Detailed log
    if (print_log)
        cout << "Matched pairs: " << matched << "/" << all << endl;

    // Deallocate memory
    for (int i = 0; i < n; ++i)
        delete[] faiss_indices[i];
    delete[] faiss_indices;

    // Return checking result
    return similarity;
}

void KnnModel::PreProcessing() {
    Clean();
    knn_indices = new int*[n];
    for (int i = 0; i < n; ++i)
        knn_indices[i] = new int[k];
    knn_distances = new float*[n];
    for (int i = 0; i < n; ++i)
        knn_distances[i] = new float[k];
    PreCalculationOfDistance();
}

void KnnModel::Solve() {
    PreProcessing();

    pair<float, int>** dist_from_to = new pair<float, int>*[n];
    for (int i = 0; i < n; ++i)
        dist_from_to[i] = new pair<float, int>[k];

    SolveForHeaps(dist_from_to);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j) {
            knn_distances[i][j] = dist_from_to[i][j].first;
            knn_indices[i][j] = dist_from_to[i][j].second;
        }

    for (int i = 0; i < n; ++i)
        delete[] dist_from_to[i];
    delete[] dist_from_to;
}

inline void KnnModel::SolveForHeaps(pair<float, int>** dist_from_to) {
    const int n_blocks = (n + block_size - 1) / block_size;
    const int last_block_size = n - (n_blocks - 1) * block_size;

    // Allocate memory
    float** blocks = new float*[n_blocks];
    for (int i = 0; i < n_blocks - 1; ++i) {
        blocks[i] = new float[block_size * d];
        const int i_base = i * block_size;
        for (int j = 0; j < block_size; ++j)
            for (int k = 0; k < d; ++k)
                blocks[i][j * d + k] = points[(i_base + j) * d + k];
    }
    {
        const int i = n_blocks - 1;
        const int i_base = i * block_size;
        blocks[i] = new float[last_block_size * d];
        for (int j = 0; j < last_block_size; ++j)
            for (int k = 0; k < d; ++k)
                blocks[i][j * d + k] = points[(i_base + j) * d + k];
    }

    float* foo = new float[block_size * block_size];

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
}

void KnnModel::PushBlockToHeap(
    const float* i_block, const int i, const int i_size,
    const float* j_block, const int j, const int j_size,
    float* sum_of_products, pair<float, int>** dist_from_to
) {
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        i_size, j_size, d, 1, i_block, d, j_block, d, 0, sum_of_products, j_size
    );

    SimdHeap heap;
    for (int ii = 0; ii < i_size; ++ii)
        for (int jj = (i < j ? 0 : ii + 1); jj < j_size; ++jj) {
            const int i_index = i * block_size + ii;
            const int j_index = j * block_size + jj;
            const float dist =
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
    sum_of_squared = new float[n];
    for (int i = 0; i < n; ++i)
        sum_of_squared[i] = 0;
    for (int i = 0, lim(n * d); i < lim; ++i)
        sum_of_squared[i / d] += points[i] * points[i];
}

void KnnModel::Clean() {
    if (knn_distances) {
        for (int i = 0; i < n; ++i)
            if (knn_distances[i]) delete[] knn_distances[i];
        delete[] knn_distances;
        knn_distances = nullptr;
    }
    if (knn_indices) {
        for (int i = 0; i < n; ++i)
            if (knn_indices[i]) delete[] knn_indices[i];
        delete[] knn_indices;
        knn_indices = nullptr;
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
    pair<float, int>* it_begin, pair<float, int>* it_end,
    int size_lim, pair<float, int> val
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

void Heap::pop_heap(pair<float, int>* it_begin, pair<float, int>* it_end) {
    swap(*it_begin, *(--it_end));
    pair<float, int> val = *it_begin;
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

void Heap::sort_heap(pair<float, int>* it_begin, pair<float, int>* it_end) {
    for (; it_end != it_begin; --it_end)
        pop_heap(it_begin, it_end);
}

void StrongHeap::push_heap(
    pair<float, int>* it_begin, pair<float, int>* it_end,
    int size_lim, pair<float, int> val
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

void StrongHeap::pop_heap(pair<float, int>* it_begin, pair<float, int>* it_end) {
    swap(*it_begin, *(--it_end));
    pair<float, int> val = *it_begin;
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

void SimdHeap::push_heap(
    pair<float, int>* it_begin, pair<float, int>* it_end,
    int size_lim, pair<float, int> val
) {
    if (it_end - it_begin == size_lim && val < *it_begin)
        pop_heap(it_begin, it_end--);
    *it_end = val;

    for (int index = it_end - it_begin + 1; index > 0;) {
        // 1. Get indices of nodes from the new node all the way up to root
        int relative_ind[8], lowest = 8;
        for (int i = 0, tmpi = (index - 1) >> 1; i < 8; ++i, tmpi = (tmpi - 1) >> 1) {
            relative_ind[i] = tmpi << 1;

            // If there are not enough 8 nodes, let the remaining be the new value
            if (tmpi == 0) {
                for (int j = i + 1; j < 8; ++j)
                    relative_ind[j] = index << 1;
                break;
            }
        }
        const __m256i indices = _mm256_load_si256((__m256i*)relative_ind);

        // 2. Get values in heap corresponding to indices
        const __m256 values = _mm256_i32gather_ps((float*)it_begin, indices, sizeof(int));

        // 3. Mask the parents that violates heap property, i.e. smaller than new value,
        //    indicating the parents that the new node needs to climb over
        const __m256 dup_new_values = _mm256_set1_ps(val.first);
        const __mmask8 cmp_mask = _mm256_cmp_ps_mask(
            values, dup_new_values,
            29 // greater-than-or-equal (ordered, non-signaling)
        );
        
        // 4. Climb to correct node
        const int n_parents_not_ge = cmp_mask != 0 ? __builtin_ctz(cmp_mask) : 8;
        
        if (n_parents_not_ge == 0) break;

        // Get pairs of data for 8 elements
        for (int i = 0; i < 8; ++i)
            relative_ind[i] >>= 1;
        
        __m256i part_1_values = _mm256_i32gather_epi64(
            (long long*)it_begin,
            _mm_load_si128((__m128i*)relative_ind),
            sizeof(int)
        );

        __m256i part_2_values = _mm256_i32gather_epi64(
            (long long*)it_begin,
            _mm_load_si128((__m128i*)(relative_ind + 4)),
            sizeof(int)
        );


        // cout << val.first << ' ' << val.second << '\n';
        // cout << it_begin[index].first << ' ' << it_begin[index].second << '\n';
        
        // Get new indices after up heap
        int up_ind = relative_ind[n_parents_not_ge - 1];
        for (int i = n_parents_not_ge - 1; i > 0; --i)
            relative_ind[i] = relative_ind[i - 1];
        relative_ind[0] = index;
        
        // Assign values to positions
        _mm256_i32scatter_epi64(
            (uint64_t*)it_begin,
            _mm_load_si128((__m128i*)relative_ind),
            part_1_values,
            sizeof(int)
        );

        _mm256_i32scatter_epi64(
            (uint64_t*)it_begin,
            _mm_load_si128((__m128i*)(relative_ind + 4)),
            part_2_values,
            sizeof(int)
        );

        // cout << it_begin[index].first << ' ' << it_begin[index].second << '\n';
        // cout << endl;

        it_begin[up_ind] = val;

        if (n_parents_not_ge != lowest || up_ind == 0) break;
        index = up_ind;
    } // end for
}