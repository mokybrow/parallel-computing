#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

void initializeMatrix(float* A, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        A[i] = (float)(rand() % 10); // инициализация случайными числами от 0 до 9
    }
}

void printMatrix(float* C, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", C[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int M = 2; // количество строк в матрице A
    int N = 3; // количество столбцов в матрице B
    int K = 4; // количество столбцов в матрице A и строк в матрице B

    // Выделяем память для матриц на хосте
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Инициализируем матрицы A и B
    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);

    // Вывод матриц
    printf("Matrix A:\n");
    printMatrix(h_A, M, K);
    printf("Matrix B:\n");
    printMatrix(h_B, K, N);

    // Выделяем память для матриц на устройстве
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // Копируем матрицы A и B на устройство
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Создаем дескриптор cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Умножаем матрицы A и B, результат сохраняется в C
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                M, N, K, 
                &alpha, 
                d_A, M,   // матрица A
                d_B, K,   // матрица B
                &beta, 
                d_C, M);  // результат C

    // Копируем результат обратно на хост
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Вывод результата
    printf("Matrix C (result of A * B):\n");
    printMatrix(h_C, M, N);

    // Освобождаем память
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}