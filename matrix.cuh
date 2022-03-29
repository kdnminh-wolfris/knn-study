#ifndef MATRIX_H
#define MATRIX_H

#include <cuda_runtime.h>
#include <stdio.h>

class Matrix {
public:
    int n_rows, n_cols;
    float* elements;

    Matrix(int r, int c);

    static Matrix mult(const Matrix A, const Matrix B);
};

#define BLOCK_SIZE 32
__global__ void MatMul(Matrix A, Matrix B, Matrix C);

#endif // MATRIX_H