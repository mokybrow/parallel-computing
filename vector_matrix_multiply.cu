#include <stdio.h>
#include <cuda.h>

__global__ void vectorMatrixMultiply(float *vec, float *mat, float *result, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        result[row] = 0.0f;
        for (int col = 0; col < N; col++) {
            result[row] += vec[col] * mat[row * N + col];
            printf("Intermediate result for row %d: vec[%d] * mat[%d][%d] = %f\n", 
                   row, col, row, col, vec[col] * mat[row * N + col]);
        }
    }
}

void printVector(float *vec, int size, const char *vecName) {
    printf("%s: ", vecName);
    for (int i = 0; i < size; i++) {
        if (i > 0) printf(", ");
        printf("%f", vec[i]);
    }
    printf("\n");
}

void printMatrix(float *mat, int M, int N, const char *matName) {
    printf("%s:\n", matName);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", mat[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    int M = 3; // Number of rows in the matrix
    int N = 4; // Number of columns in the matrix
    
    size_t vecSize = N * sizeof(float);
    size_t matSize = M * N * sizeof(float);
    size_t resultSize = M * sizeof(float);
    
    // Allocate memory on the host
    float *h_vec = (float*)malloc(vecSize);
    float *h_mat = (float*)malloc(matSize);
    float *h_result = (float*)malloc(resultSize);

    // Initialize vector and matrix
    for (int i = 0; i < N; i++) {
        h_vec[i] = 1.0f + i; // Example vector
    }
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_mat[i * N + j] = 1.0f + i + j; // Example matrix
        }
    }
    
    // Print vector and matrix
    printVector(h_vec, N, "Vector");
    printMatrix(h_mat, M, N, "Matrix");

    // Allocate memory on the device
    float *d_vec, *d_mat, *d_result;
    cudaMalloc((void**)&d_vec, vecSize);
    cudaMalloc((void**)&d_mat, matSize);
    cudaMalloc((void**)&d_result, resultSize);

    // Copy data from host to device
    cudaMemcpy(d_vec, h_vec, vecSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, h_mat, matSize, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (M + blockSize - 1) / blockSize;
    vectorMatrixMultiply<<<numBlocks, blockSize>>>(d_vec, d_mat, d_result, M, N);

    // Copy result back to host
    cudaMemcpy(h_result, d_result, resultSize, cudaMemcpyDeviceToHost);

    // Print result
    printf("Result of vector-matrix multiplication:\n");
    printVector(h_result, M, "Result Vector");

    // Free memory
    free(h_vec);
    free(h_mat);
    free(h_result);
    cudaFree(d_vec);
    cudaFree(d_mat);
    cudaFree(d_result);

    return 0;
}