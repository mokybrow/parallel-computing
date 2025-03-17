#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define N 3 // Размер вектора и матрицы
#define EPS 1e-3 // Точность
#define TAU 0.01 // Параметр метода

// CUDA Kernel для вычисления следующего приближения
__global__ void computeNextApproximation(float *A, float *b, float *xn, float *xn1) {
    int index = threadIdx.x;
    if (index < N) {
        float Ax = 0.0;
        for (int j = 0; j < N; j++) {
            Ax += A[index * N + j] * xn[j]; // Вычисление A*x
        }
        xn1[index] = xn[index] - TAU * (Ax - b[index]); // Обновление следующего приближения
    }
}

int main() {
    // Объявление и инициализация массивов
    float h_A[N * N]; // матрица A
    float h_b[N]; // вектор b
    float h_xn[N] = {0.0f}; // начальное приближение (вектор из нулей)
    float h_xn1[N]; // следующее приближение

    // Инициализация матрицы A как единичной и вектора b
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = (i == j) ? 1.0f : 0.0f; // оригинальная единичная матрица
        }
        h_b[i] = 1.0f; // Вектор b, состоящий из единиц
    }

    float *d_A, *d_b, *d_xn, *d_xn1;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_xn, N * sizeof(float));
    cudaMalloc((void**)&d_xn1, N * sizeof(float));

    // Копирование данных на устройство
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xn, h_xn, N * sizeof(float), cudaMemcpyHostToDevice);

    float eps = 1e6; // Начальная невязка
    int num_iter = 0;

    while (eps > EPS) { // Цикл до достижения нужной точности
        // Вызов ядра
        computeNextApproximation<<<1, N>>>(d_A, d_b, d_xn, d_xn1);
        cudaDeviceSynchronize();

        // Копирование следующего приближения с устройства на хост
        cudaMemcpy(h_xn1, d_xn1, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Вычисление невязки
        eps = 0.0f;
        for (int i = 0; i < N; i++) {
            eps = fmaxf(eps, fabsf(h_xn1[i] - h_xn[i])); // Максимальный модуль разности
        }

        // Обновление текущего приближения на хосте
        for (int i = 0; i < N; i++) {
            h_xn[i] = h_xn1[i]; // Используйте новое приближение для следующей итерации
        }

        // Копирование нового приближения с хоста на устройство
        cudaMemcpy(d_xn, h_xn, N * sizeof(float), cudaMemcpyHostToDevice);
        num_iter++;

        printf("Iter: %d, EPS: %f\n", num_iter, eps);
    }

    // Копирование финального результата на хост
    cudaMemcpy(h_xn, d_xn, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Печать результата
    printf("Final solution:\n");
    for (int i = 0; i < N; i++) {
        printf("x[%d] = %f\n", i, h_xn[i]);
    }

    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_xn);
    cudaFree(d_xn1);

    return 0;
}