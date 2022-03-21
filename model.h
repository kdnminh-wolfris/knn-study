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
    double* points = nullptr; // array of data points

    // size of each block for processing matrix multiplication
    int block_size = 1000;

    // sum of squared points[i][j] for pre-calculation of distances
    double* sum_of_squared = nullptr;

    // list of indexes of k nearest neighbours for each data point
    int** results = nullptr;

    void PreProcessing();    
    void PreCalculationOfDistance();

    void SolveForHeaps(pair<double, int>** heap);
    void PushBlockToHeap(
        const double* i_block, const int i, const int i_size,
        const double* j_block, const int j, const int j_size,
        double* sum_of_products, pair<double, int>** heap
    );
};

/**
 * @brief Given a max heap in the range [it_begin, it_begin + heap_size), this function
 * maintains the size of the heap by pushing the new value val into heap only when it's
 * less than the current heap top.
 * 
 * @param it_begin the initial position of the heap
 * @param heap_size the number of elements in the heap
 * @param val the value to be pushed into heap
 */
void push_heap(pair<double, int>* it_begin, int heap_size, pair<double, int> val);

/**
 * @brief Rearranges the elements in the heap range [it_begin, it_end) in such a way
 * that the part considered a heap is shortened by one: The element with the highest
 * value is moved to (it_end-1).
 * 
 * @param it_begin the initial position of the heap
 * @param it_end the final position of the heap
 */
void pop_heap(pair<double, int>* it_begin, pair<double, int>* it_end);

/**
 * @brief Sorts the elements in the heap range [it_begin, it_end) into ascending order.
 * 
 * @param it_begin the initial position of the heap
 * @param it_end the final position of the heap
 */
void sort_heap(pair<double, int>* it_begin, pair<double, int>* it_end);

void make_heap(pair<double, int>* it_begin, pair<double, int>* it_end);

#endif // KNN_MODEL