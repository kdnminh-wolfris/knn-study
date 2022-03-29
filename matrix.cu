#include "matrix.cuh"

Matrix::Matrix(int r, int c) : n_rows(r), n_cols(c) {}

Matrix Matrix::mult(const Matrix A, const Matrix B) {
    // Allocate memory for matrix A on device
    Matrix dA(A.n_rows, A.n_cols);
    cudaMalloc(&dA.elements, dA.n_rows * dA.n_cols * sizeof(float));
    cudaMemcpy(
        dA.elements, A.elements,
        dA.n_rows * dA.n_cols * sizeof(float),
        cudaMemcpyHostToDevice
    );

    // Allocate memory for matrix B on device
    Matrix dB(B.n_rows, B.n_cols);
    cudaMalloc(&dB.elements, dB.n_rows * dB.n_cols * sizeof(float));
    cudaMemcpy(
        dB.elements, B.elements,
        dB.n_rows * dB.n_cols * sizeof(float),
        cudaMemcpyHostToDevice
    );

    // Initiate matrix C and allocate its memory on device
    Matrix C(A.n_rows, B.n_cols), dC(A.n_rows, B.n_cols);
    C.elements = new float[C.n_rows * C.n_cols];
    cudaMalloc(&dC.elements, dC.n_rows * dC.n_cols * sizeof(float));

    // Do matrix multiplication on GPU
    const int n_block_rows = (A.n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int n_block_cols = (B.n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    MatMul<<<
        dim3(n_block_rows, n_block_cols),
        dim3(BLOCK_SIZE, BLOCK_SIZE)
    >>>(dA, dB, dC);

    // Copy result for matrix C from device to host
    cudaMemcpy(
        C.elements, dC.elements,
        C.n_rows * C.n_cols * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // Deallocate device memory
    cudaFree(dA.elements);
    cudaFree(dB.elements);
    cudaFree(dC.elements);

    // Return results
    return C;
}

__global__ void MatMul(Matrix A, Matrix B, Matrix C) {
    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;

    const int subI = threadIdx.x;
    const int subJ = threadIdx.y;

    const int i = blockRow * BLOCK_SIZE + subI;
    const int j = blockCol * BLOCK_SIZE + subJ;
    const bool flag = i < C.n_rows && j < C.n_cols;

    const int n_block_k = (A.n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float Cval = 0;
    
    for (int blockK = 0; blockK < n_block_k; ++blockK) {
        __shared__ float subA[BLOCK_SIZE * BLOCK_SIZE]; // row-major
        __shared__ float subB[BLOCK_SIZE * BLOCK_SIZE]; // col-major

        const int k = blockK * BLOCK_SIZE;

        if (i < A.n_rows && k + subJ < A.n_cols)
            subA[subI * BLOCK_SIZE + subJ] = A.elements[i * A.n_cols + (k + subJ)];
        
        if (k + subI < B.n_rows && j < B.n_cols)
            subB[subI + subJ * BLOCK_SIZE] = B.elements[(k + subI) * B.n_cols + j];

        __syncthreads();

        if (flag)
            for (int subK = min(BLOCK_SIZE, A.n_cols - blockK * BLOCK_SIZE) - 1; subK >= 0; --subK)
                Cval += subA[subI * BLOCK_SIZE + subK] * subB[subK + subJ * BLOCK_SIZE];

        __syncthreads();
    }

    if (flag) C.elements[i * C.n_cols + j] = Cval;
}