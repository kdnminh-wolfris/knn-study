#include <iostream>
#include "matrix.cuh"

using namespace std;

int main() {
    Matrix A(4, 2);
    Matrix B(2, 4);

    A.elements = new float[8];
    B.elements = new float[8];

    for (int i = 0; i < 8; ++i)
        A.elements[i] = B.elements[i] = 1;

    Matrix C = Matrix::mult(A, B);

    for (int i = 0; i < C.n_rows; ++i)
        for (int j = 0; j < C.n_cols; ++j)
            cout << C.elements[i * C.n_rows + j] << " \n"[j == C.n_cols - 1];
    
    delete[] A.elements;
    delete[] B.elements;
    delete[] C.elements;
}