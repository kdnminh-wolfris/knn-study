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
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

/**
 * @brief List of knn algorithms to use
 * 
 */
enum Algorithm {
    ver1, ver1_2, ver2, ver3
};

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
     * @brief Set knn algorithm for model.
     * 
     * @param algo knn algorithm
     */
    void SetAlgorithm(Algorithm algo);

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

private:
    int n; // number of data points
    int d; // number of dimensions
    int k; // number of nearest neighbours to find

    MatrixXd points; // array of data points
    // vector<vector<double>> points;

    Algorithm algorithm; // current algorithm to solve

    vector<vector<int>> results; // list of indexes of k nearest neighbours for each data point
    
    vector<double> sum_of_squared; // sum of squared points[i][j] for pre-calculation of distances

    /**
     * @brief Naive O(N^2 * d) solution
     * 
     */
    void _SolveVer1();

    /**
     * @brief Naive solution with symmetrically calculating and storing distances, O(N(N - 1)/2 * d)
     * 
     */
    void _SolveVer1_2();

    /**
     * @brief Using heap structure to store exactly k nearest neighbours for each point
     * 
     */
    void _SolveVer2();

    /**
     * @brief Dividing the data matrix into blocks and performing matrix multiplication by using Eigen library
     * and storing results using heap structures
     * 
     */
    void _SolveVer3();

    /**
     * @brief Calculate the Euclidean distance between points indexed i and j.
     * 
     * @param a first data point
     * @param b second data point
     * @return the distance between data points
     */
    double Distance(int i, int j);
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