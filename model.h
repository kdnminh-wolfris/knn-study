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
#include <immintrin.h>

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

    float SimilarityCheck(string indices_path, bool print_log);

    void Clean();

    ~KnnModel();

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

    void SolveForHeaps(pair<float, int>** dist_from_to);
    void PushBlockToHeap(
        const float* i_block, const int i, const int i_size,
        const float* j_block, const int j, const int j_size,
        float* sum_of_products, pair<float, int>** dist_from_to
    );
};

class Heap {
public:
    /**
     * @brief Given a max heap in the range [it_begin, it_end), this function extends
     * the range considered a heap to [it_begin, it_end] by placing val in it_end into
     * its corresponding location within it. The heap is also kept not to exceed
     * size_lim elements by popping the heap top before pushing the new value val.
     * 
     * @param it_begin the initial position of the heap
     * @param it_end the final position of the heap
     * @param size_lim the number of elements in the heap is limited by size_lim
     * @param val the value to be pushed into heap
     */
    void push_heap(
        pair<float, int>* it_begin, pair<float, int>* it_end,
        int size_lim, pair<float, int> val
    );

    /**
     * @brief Rearranges the elements in the heap range [it_begin, it_end) in such a way
     * that the part considered a heap is shortened by one: The element with the highest
     * value is moved to (it_end-1).
     * 
     * @param it_begin the initial position of the heap
     * @param it_end the final position of the heap
     */
    void pop_heap(pair<float, int>* it_begin, pair<float, int>* it_end);

    /**
     * @brief Sorts the elements in the heap range [it_begin, it_end) into ascending
     * order.
     * 
     * @param it_begin the initial position of the heap
     * @param it_end the final position of the heap
     */
    void sort_heap(pair<float, int>* it_begin, pair<float, int>* it_end);
};

class StrongHeap : Heap {
public:
    void push_heap(
        pair<float, int>* it_begin, pair<float, int>* it_end,
        int size_lim, pair<float, int> val
    );
    void pop_heap(pair<float, int>* it_begin, pair<float, int>* it_end);
};

class SimdHeap : Heap {
public:
    void push_heap(
        pair<float, int>* it_begin, pair<float, int>* it_end,
        int size_lim, pair<float, int> val
    );
};

#endif // KNN_MODEL