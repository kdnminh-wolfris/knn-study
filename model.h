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

#include <iostream>
#include <vector>

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

private:
    int n; // number of data points
    int d; // number of dimensions
    int k; // number of nearest neighbours to find

    int block_size = 500; // size of each block for processing matrix multiplication

    double* points; // array of data points

    vector<vector<int>> results; // list of indexes of k nearest neighbours for each data point
    
    vector<double> sum_of_squared; // sum of squared points[i][j] for pre-calculation of distances

    void PreProcessing();    
    void PreCalculationOfDistance();
};

template<typename T>
void push_heap(T* it_begin, T* it_end);

template<typename T>
void pop_heap(T* it_begin, T* it_end);

template<typename T>
void sort_heap(T* it_begin, T* it_end);

template<typename T>
void push_heap(T* it_begin, T* it_end, int size_lim, T val);

#endif // KNN_MODEL