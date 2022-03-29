#ifndef EXACT_SOLVER
#define EXACT_SOLVER

#include <string>

using namespace std;

class KnnModel {
private:
    int n; // Number of points
    int d; // Number of dimensions
    int k; // Number of neighbours to find

    // An n×d matrix indicating data of n points with in d-dimension space
    float* points = nullptr;

    // An n×k matrix indicating indices of k nearest neighbours to each of n points
    int* res_indices = nullptr;

    // An n×k matrix indicating distances of k nearest neighbours to each of n points,
    // corresponding to indices in res_indices
    float* res_distances = nullptr;

public:
    /**
     * @brief Reads data of an instance of k nearest neighbours problem from a file
     * 
     * @param path Path to input file
     */
    void ReadData(string path);

    /**
     * @brief Solves the k nearest neighbours problem with the current instance
     * 
     */
    void Solve();

    /**
     * @brief Writes out the results of indices and distances of k nearest neighbours to
     * files indices.out and distances.out, respectively, in the given directory
     * 
     * @param path Path to the directory for writing out results
     */
    void WriteResults(string path);

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
};

#endif // EXACT_SOLVER