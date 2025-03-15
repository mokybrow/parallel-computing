#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void update_xn(float* xn, float* A, float* b, float tau, int n, float* eps) {
    int idx = threadIdx.x;

    // Расчет следующего приближения
    if (idx < n) {
        float Axn = 0.0f;
        for (int j = 0; j < n; j++) {
            Axn += A[idx * n + j] * xn[j];
        }
        float new_xn = xn[idx] - tau * (Axn - b[idx]);
        
        // Сохраняя новое значение в xn
        xn[idx] = new_xn;

        // Вычисление epsilon только для одного потока
        if (idx == 0) {
            *eps = 0.0f;
            for (int j = 0; j < n; j++) {
                float diff = fabs(new_xn - xn[j]); // Используем новое значение для расчета
                if (diff > *eps) {
                    *eps = diff;
                }
            }
        }
    }
}

int main() {
    int n = 3; // размер вектора
    float h_xn[3] = {0.0f, 0.0f, 0.0f}; // начальное приближение
    float h_A[9] = {1.0f, 0.0f, 0.0f, 
                    0.0f, 1.0f, 0.0f, 
                    0.0f, 0.0f, 1.0f}; // единичная матрица
    float h_b[3];
    
    // вычисление b как A * x = [1, 1, 1]
    for (int i = 0; i < n; i++) {
        h_b[i] = 1.0f; // Вектор b - все элементы равны 1
    }

    float *d_xn, *d_A, *d_b, *d_eps;
    float h_eps = 1e6;
    float tau = 0.01;
    int num_iter = 0;

    cudaMalloc(&d_xn, n * sizeof(float));
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_eps, sizeof(float));

    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xn, h_xn, n * sizeof(float), cudaMemcpyHostToDevice);
    
    while (h_eps > 1e-3) {
        cudaMemcpy(d_eps, &h_eps, sizeof(float), cudaMemcpyHostToDevice);
        
        update_xn<<<1, n>>>(d_xn, d_A, d_b, tau, n, d_eps);
        
        cudaMemcpy(h_xn, d_xn, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_eps, d_eps, sizeof(float), cudaMemcpyDeviceToHost);
        
        num_iter++;
        printf("Iteration %d: eps = %f\n", num_iter, h_eps);
    }

    printf("Solution: [");
    for (int i = 0; i < n; i++) {
        printf("%f", h_xn[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");

    cudaFree(d_xn);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_eps);
    
    return 0;
}