#include <stdio.h>
#include <cuda.h>

#define N 10090000 // Размер векторов

// CUDA Kernel для сложения двух векторов
__global__ void addVectors(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Индекс текущего потока
    if (i < n) {
        C[i] = A[i] + B[i]; // Сложение векторов
    }
}

int main() {
    // Выделение памяти для векторов на хосте
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    // Инициализация векторов
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f; // Пример заполнения первого вектора
        h_B[i] = i * 2.0f; // Пример заполнения второго вектора
    }


    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Копирование данных из хоста на устройство
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Определение размеров блоков и сетки
    int threadsPerBlock = 256; // Количество потоков на блок
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // Количество блоков

    // Вызов ядра
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Копирование результата с устройства на хост
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Вывод результата (например, первых 10 элементов)
    printf("Result of A + B:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
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