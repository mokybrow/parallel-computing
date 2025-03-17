#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 3 // Размер матриц

void printMatrix(const float* matrix, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Размеры матриц
    int rowsA = N, colsA = N;
    int rowsB = N, colsB = N;
    int rowsC = rowsA, colsC = colsB;

    // Создание и инициализация матриц на хосте
    float h_A[N * N] = {1, 2, 3, 4, 5, 6, 7, 8, 9}; // Матрица A
    float h_B[N * N] = {9, 8, 7, 6, 5, 4, 3, 2, 1}; // Матрица B
    float h_C[N * N]; // Результат C

    // Указатели на устройства
    float *d_A, *d_B, *d_C;

    // Выделение памяти на устройстве
    cudaMalloc((void**)&d_A, rowsA * colsA * sizeof(float));
    cudaMalloc((void**)&d_B, rowsB * colsB * sizeof(float));
    cudaMalloc((void**)&d_C, rowsC * colsC * sizeof(float));

    // Копирование матриц A и B на устройство
    cudaMemcpy(d_A, h_A, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);

    // Инициализация CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Параметры для функции cublasSgemm
    float alpha = 1.0f; // Скаляры
    float beta = 0.0f;

    // Умножение матриц C = A * B
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                rowsC, colsC, colsA,
                &alpha, d_A, rowsA, // матрица A
                d_B, colsB,         // матрица B
                &beta, d_C, rowsC); // результирующая матрица C

    // Копирование результата обратно на хост
    cudaMemcpy(h_C, d_C, rowsC * colsC * sizeof(float), cudaMemcpyDeviceToHost);

    // Печать результата
    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_A, rowsA, colsA);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_B, rowsB, colsB);
    std::cout << "Result Matrix C (A * B):" << std::endl;
    printMatrix(h_C, rowsC, colsC);

    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}