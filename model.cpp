#include <fstream>
#include <algorithm>
#include <math.h>
#include <thread>

#include "model.h"

KnnModel::KnnModel() {
    // intentionally left blank
}

bool KnnModel::ReadData(string path) {
    // Simple case first
    ifstream fi(path);
    fi >> n >> d >> k;
    // n = 1000;
    
    points.resize(n, d);
    // points.resize(n, vector<double>(d));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j)
            // fi >> points[i][j];
            fi >> points(i, j);
    fi.close();
    return true;
}

void KnnModel::SetAlgorithm(Algorithm algo) {
    algorithm = algo;
}

void KnnModel::Solve() {
    results.resize(n, vector<int>(k, -1));
    PreCalculationOfDistance();
    switch (algorithm) {
        case ver1:
            _SolveVer1();
            break;
        case ver1_2:
            _SolveVer1_2();
            break;
        case ver2:
            _SolveVer2();
            break;
        case ver3:
            _SolveVer3();
            break;
        default:
            break;
    }
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

void KnnModel::_SolveVer1() {
    pair<double, int>* distance_to = new pair<double, int>[n];
    for (int i = 0; i < n; ++i) {
        distance_to[i] = {-1, i};
        for (int j = 0; j < n; ++j)
            if (i != j)
                distance_to[j] = {Distance(i, j), j};
        sort(distance_to, distance_to + n);
        for (int j = 1; j <= k; ++j)
            results[i][j - 1] = distance_to[j].second;
    }
    delete[] distance_to;
}

void KnnModel::_SolveVer1_2() {
    pair<double, int>** distance_to = new pair<double, int>*[n];
    for (int i = 0; i < n; ++i)
        distance_to[i] = new pair<double, int>[n];
    for (int i = 0; i < n; ++i) {
        distance_to[i][i] = {-1, i};
        for (int j = i + 1; j < n; ++j) {
            double dist = Distance(i, j);
            distance_to[i][j] = {dist, j};
            distance_to[j][i] = {dist, i};
        }
        sort(distance_to[i], distance_to[i] + n);
        for (int j = 1; j <= k; ++j)
            results[i][j - 1] = distance_to[i][j].second;
    }
    for (int i = 0; i < n; ++i)
        delete[] distance_to[i];
    delete[] distance_to;
}

void KnnModel::_SolveVer2() {
    pair<double, int>** dist_from_to = new pair<double, int>*[n];
    for (int i = 0; i < n; ++i)
        dist_from_to[i] = new pair<double, int>[k];
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double dist(Distance(i, j));
            
            // insert heap[i]
            if (j <= k || dist < dist_from_to[i][0].first) {
                if (j > k)
                    pop_heap(dist_from_to[i], dist_from_to[i] + k);
                dist_from_to[i][min(j, k) - 1] = {dist, j};
                push_heap(dist_from_to[i], dist_from_to[i] + min(j, k));
            }

            // insert heap[j]
            if (i < k || dist < dist_from_to[j][0].first) {
                if (i >= k)
                    pop_heap(dist_from_to[j], dist_from_to[j] + k);
                dist_from_to[j][min(i, k - 1)] = {dist, i};
                push_heap(dist_from_to[j], dist_from_to[j] + min(i + 1, k));
            }
        }

        // get results for point i
        sort_heap(dist_from_to[i], dist_from_to[i] + k);
        for (int j = 0; j < k; ++j)
            results[i][j] = dist_from_to[i][j].second;
    }
    for (int i = 0; i < n; ++i)
        delete[] dist_from_to[i];
    delete[] dist_from_to;
}

void KnnModel::_SolveVer3() {
    // ofstream fo("debug.out");

    pair<double, int>** dist_from_to = new pair<double, int>*[n];
    for (int i = 0; i < n; ++i)
        dist_from_to[i] = new pair<double, int>[k];

    cout << "Done allocating memory for heaps" << endl;

    const int block_size = 500;
    const int n_blocks = (n + block_size - 1) / block_size;
    vector<MatrixXd> blocks(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        blocks[i].resize(0, d);
        for (int j = 0, lim = min(block_size, n - i * block_size); j < lim; ++j) {
            blocks[i].conservativeResize(blocks[i].rows() + 1, blocks[i].cols());
            blocks[i].row(blocks[i].rows() - 1) = points.row(i * block_size + j);
        }
    }

    cout << "Done creating blocks" << endl;

    for (int i = 0; i < n_blocks; ++i) {
        cout << "i = " << i << ' ';
        auto start = chrono::high_resolution_clock::now();

        for (int j = i; j < n_blocks; ++j) {
            MatrixXd foo = blocks[i] * blocks[j].transpose();
            
            for (int ii = 0; ii < foo.rows(); ++ii)
                for (int jj = (i < j ? 0 : ii + 1); jj < foo.cols(); ++jj) {
                    int i_index = i * block_size + ii;
                    int j_index = j * block_size + jj;
                    double dist = sum_of_squared[i_index] + sum_of_squared[j_index] - 2 * foo(ii, jj);

                    // insert heap[i]
                    if (j_index <= k || dist < dist_from_to[i_index][0].first)
                        push_heap(dist_from_to[i_index], dist_from_to[i_index] + min(j_index - 1, k), k, {dist, j_index});

                    // insert heap[j]
                    if (i_index < k || dist < dist_from_to[j_index][0].first)
                        push_heap(dist_from_to[j_index], dist_from_to[j_index] + min(i_index, k), k, {dist, i_index});
                }
        }
        
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << duration.count() / 1000000 << '.' << duration.count() % 1000000 << 's' << endl;
    }

    cout << "Done solving" << endl;

    for (int i = 0; i < n; ++i) {
        sort_heap(dist_from_to[i], dist_from_to[i] + k);
        for (int j = 0; j < k; ++j)
            results[i][j] = dist_from_to[i][j].second;
    }

    cout << "Done getting results" << endl;

    for (int i = 0; i < n; ++i)
        delete[] dist_from_to[i];
    delete[] dist_from_to;

    cout << "Done unallocating memory" << endl;

    // fo.close();
}


double KnnModel::Distance(int i, int j) {
    // double s(0);
    // for (int x = 0; x < d; ++x)
    //     s += points[i][x] * points[j][x];
    // return sum_of_squared[i] + sum_of_squared[j] - 2 * s;

    return (points.row(i) - points.row(j)).norm();

    // return sum_of_squared[i] + sum_of_squared[j] - 2 * psum(i, j);
}

void KnnModel::PreCalculationOfDistance() {
    sum_of_squared.resize(n, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j)
            // sum_of_squared[i] += points[i][j] * points[i][j];
            sum_of_squared[i] += points(i, j) * points(i, j);
    
    // psum = points * points.transpose();
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