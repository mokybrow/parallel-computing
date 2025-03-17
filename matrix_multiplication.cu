#include <stdio.h>
#include <cuda.h>

#define MATRIX_SIZE 3 // Размер матриц (MATRIX_SIZE x MATRIX_SIZE)

// CUDA Kernel для умножения матриц
__global__ void multiplyMatrices(float *A, float *B, float *C, int matrixSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Индекс текущего потока (номер строки результата)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Индекс текущего потока (номер столбца результата)

    if (row < matrixSize && col < matrixSize) {
        float sum = 0.0f;
        for (int k = 0; k < matrixSize; k++) {
            float product = A[row * matrixSize + k] * B[k * matrixSize + col]; // Произведение элементов матриц
            // printf("Multiplying A[%d][%d] * B[%d][%d]: %f * %f = %f\n", row, k, k, col, A[row * matrixSize + k], B[k * matrixSize + col], product);
            sum += product; // Суммируем произведения
        }
        C[row * matrixSize + col] = sum; // Записываем результат в матрицу C
    }
}

int main() {
    // Задаем размер матриц
    int matrixSize = MATRIX_SIZE;

    // Выделение памяти для матриц и результата на хосте
    float *h_A = (float*)malloc(matrixSize * matrixSize * sizeof(float));
    float *h_B = (float*)malloc(matrixSize * matrixSize * sizeof(float));
    float *h_C = (float*)malloc(matrixSize * matrixSize * sizeof(float));

    // Инициализация матрицы A и B
    for (int i = 0; i < matrixSize * matrixSize; i++) {
        h_A[i] = i + 1.0f; // Пример заполнения матрицы A
        h_B[i] = (i % MATRIX_SIZE) + 1.0f; // Пример заполнения матрицы B
    }

    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize * matrixSize * sizeof(float));
    cudaMalloc((void**)&d_B, matrixSize * matrixSize * sizeof(float));
    cudaMalloc((void**)&d_C, matrixSize * matrixSize * sizeof(float));

    // Копирование данных из хоста на устройство
    cudaMemcpy(d_A, h_A, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);

    // Определение размеров блоков и сетки
    dim3 threadsPerBlock(1, 1); // Количество потоков на блок
    dim3 blocksPerGrid(matrixSize, matrixSize); // Количество блоков

    // Вызов ядра
    multiplyMatrices<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, matrixSize);
    
    // Синхронизация потоков, чтобы убедиться, что все выводы были выполнены
    cudaDeviceSynchronize();

    // Копирование результата с устройства на хост
    cudaMemcpy(h_C, d_C, matrixSize * matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Печать результата
    printf("Result of matrix multiplication (C = A * B):\n");
    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            printf("%f ", h_C[i * matrixSize + j]);
        }
        printf("\n");
    }

    // Освобождение памяти
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}