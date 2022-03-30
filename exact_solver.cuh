#ifndef EXACT_SOLVER
#define EXACT_SOLVER

#include "model.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

class KnnSolver : public KnnModel {
protected:
    // Matrix of points on device
    float* d_points;
    
    // n-dimension vector of sums of squared dimension values for each point on device
    float* sum_of_sqr;

    // An n×k row-major matrix indicating indices of k nearest neighbours to each of n
    // points
    int* res_indices = nullptr;

    // An n×k row-major matrix indicating distances of k nearest neighbours to each of
    // n points, corresponding to indices in res_indices
    float* res_distances = nullptr;

    /**
     * @brief Initialises some values to boost the solving process
     * 
     */
    inline void PreProcessing();

    /**
     * @brief Initialises result variables of indices and distances
     * 
     */
    inline void ResultInit();

    inline void __Solve();
    void CalcSortMerge(
        const int i, const float* i_block, const int i_size,
        const int j, const float* j_block, const int j_size,
        float* inner_prod, cublasHandle_t handle,
        pair<float, int>* foo, pair<float, int>* d_foo
    );

    /**
     * @brief Cleans the current instance of the solver
     * 
     */
    void Clean();

    void CleanOnDevice();

public:
    /**
     * @brief Solves the k nearest neighbours problem with the current instance
     * 
     */
    void Solve();

    /**
     * @brief Checks the percentage of similar neighbour indices found of the solver
     * results and Faiss' results
     * 
     * @param path Path to the directory that contains both Faiss' result files named
     * faiss_indices.out and faiss_distances.out
     * @param print_log Determines to or not to print out the detailed log when checking
     * results to console
     * @return The percentage of similar neighbour indices found of the solver results
     * and Faiss' results
     */
    float SimilarityCheck(string path, bool print_log = true);

    /**
     * @brief Writes out the results of indices and distances of k nearest neighbours to
     * files indices.out and distances.out, respectively, in the given directory
     * 
     * @param path Path to the directory for writing out results
     */
    void WriteResults(string path);

    ~KnnSolver();
};

__global__ void CalculateSumOfSquared(
    const int n, const int d, const float* points, float* sum_of_sqr
);

void Sort(pair<float, int>* first, pair<float, int>* last, int k);

#endif // EXACT_SOLVER