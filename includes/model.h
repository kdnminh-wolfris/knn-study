#ifndef __KNN_MODEL__
#define __KNN_MODEL__

#include "config.h"
#include <string>

using namespace std;

class KnnModel {
protected:
    int n; // Number of points
    int d; // Number of dimensions
    int k; // Number of neighbours to find

    // An n×d row-major matrix indicating data of n points with in d-dimension space
    float* points = nullptr;

    /**
     * @brief Cleans the current instance of the model
     * 
     */
    void Clean();

public:
    /**
     * @brief Reads data of an instance of k nearest neighbours problem from a file
     * 
     * @param path Path to input file
     */
    void ReadData(const string path);

    ~KnnModel();
};

#endif // KNN_MODEL