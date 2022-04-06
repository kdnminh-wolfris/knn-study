#ifndef __EXACT_SOLVER__
#define __EXACT_SOLVER__

#include "model.h"
#include "config.h"
#include "utils.h"
#include "kernel.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

class KnnSolver : public KnnModel {
private:
    cublasHandle_t handle;
    
    // cuBLAS matrix multiplication coefficients
    const float alpha = 1;
    const float beta = 0;
    
    float* inner_prod;

    float* dist;
    int* ind;
    
    float* tdist;
    int* tind;

    cub::DoubleBuffer<float> db_dist;
    cub::DoubleBuffer<int> db_ind;
    
    float* tmp_d;
    int* tmp_i;
    float* dtmp_d;
    int* dtmp_i;

    void* aux = nullptr;
    size_t aux_size;
    size_t pre_aux_size = 0;

    cub::CountingInputIterator<int> itr = cub::CountingInputIterator<int>(0);

    inline void __Solve();

    /**
     * @brief Initiates values and allocates auxiliary memory
     * 
     */
    inline void __PreProcessing();

    /**
     * @brief Deallocates auxiliary memory
     * 
     */
    inline void __PostProcessing();

    /**
     * @brief Calculates distances of each pair of points in blocks i and j, and assigns
     * them to db_dist.Current() along with corresponding indices in db_ind.Current()
     * 
     */
    inline void __Calc(
        const int i_size, const float *i_block,
        const int j, const int j_size, const float *j_block
    );
    
    /**
     * @brief Sorts neighbours in block j of each point in block i (stored in
     * db_dist.Current() and db_ind.Current()) by their distances
     * 
     */
    inline void __Sort(const int i_size, const int j_size);

    /**
     * @brief Merges the current k nearest neighbours of each point with the neighbours
     * which are just calculated and sorted in db_dist.Current() and db_ind.Current(),
     * and keep the k nearest in the result arrays d_distances and d_indices
     * 
     */
    inline void __Merge(
        const int i, const int i_size,
        const int j, const int j_size
    );

protected:
    // An n×k row-major matrix indicating indices of k nearest neighbours to each of n
    // points
    int* res_indices = nullptr;

    // An n×k row-major matrix indicating distances of k nearest neighbours to each of
    // n points, corresponding to indices in res_indices
    float* res_distances = nullptr;

    // Matrix of points on device
    float* d_points;
    
    // n-dimension vector of sums of squared dimension values for each point on device
    float* sum_of_sqr;

    int* d_indices;
    float* d_distances;

    /**
     * @brief Initialises some values to boost the solving process
     * 
     */
    inline void PreProcessing();

    inline void PostProcessing();

    /**
     * @brief Initialises result variables of indices and distances
     * 
     */
    inline void ResultInit();

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

#endif // EXACT_SOLVER