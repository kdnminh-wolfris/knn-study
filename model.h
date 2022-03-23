/**
 * @file model.h
 * @author Khau Dang Nhat Minh
 * @brief Provide model for k nearest neighbours problem
 * @version 0.1
 * @date 2022-03-04
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef KNN_MODEL
#define KNN_MODEL

#include <string>
#include <x86intrin.h>
#include <cstdint>
#include <assert.h>

using namespace std;

class KnnModel {
public:
    KnnModel();

    /**
     * @brief Input an instance of knn problem from a file.
     * 
     * @param path path to input file
     * @return true if data was successfully inputed, false otherwise.
     */
    bool ReadData(string path);

    /**
     * @brief Execute the current algorithm to solve knn problem.
     * 
     */
    void Solve();

    /**
     * @brief Output result to a file.
     * 
     */
    void Output(string path);

    void Clean();

    ~KnnModel();

    long long timecnt = 0;

private:
    int n = 0; // number of data points
    int d = 0; // number of dimensions
    int k = 0; // number of nearest neighbours to find
    float* points = nullptr; // array of data points

    // size of each block for processing matrix multiplication
    int block_size = 5000;

    // sum of squared points[i][j] for pre-calculation of distances
    float* sum_of_squared = nullptr;

    // list of distances to k nearest neighbours corresponding to indices for each data point
    float** knn_distances = nullptr;
    
    // list of indexes of k nearest neighbours for each data point
    int** knn_indices = nullptr;

    void PreProcessing();    
    void PreCalculationOfDistance();

    void GetResults(pair<float, int>** dist_from_to);
    void CalcSortMerge(
        const float* i_block, const int i, const int i_size,
        const float* j_block, const int j, const int j_size,
        float* sum_of_products, pair<float, int>** dist_from_to
    );
};

template<typename T>
void selectionsort(T* first, T* last, int k);

template<typename T>
void down_reverse_heap(T* first, T* last, int index);

template<typename T>
void pop_reverse_heap(T* first, T* last);

template<typename T>
void make_reverse_heap(T* first, T* last);

template<typename T>
void heapsort(T* first, T* last, int k);

template<typename T>
T* choose_pivot(T* first, T* last);

template<typename T>
T* partition(T* first, T* last);

/**
 * @brief Sorts the first k elements in range [first, last).
 * 
 * @param first the initial iterator
 * @param last the final interator
 * @param k number of elements in the beginning of the array to be sorted
 */
template<typename T>
void sort(T* first, T* last, int k);

#endif // KNN_MODEL