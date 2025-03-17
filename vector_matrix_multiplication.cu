#include <stdio.h>
#include <cuda.h>

#define VECTOR_SIZE 3 // Размер вектора
#define MATRIX_SIZE 3 // Размер матрицы (MATRIX_SIZE x MATRIX_SIZE)

// CUDA Kernel для умножения вектора на матрицу
__global__ void multiplyVectorByMatrix(float *vector, float *matrix, float *result, int vectorSize, int matrixSize) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Индекс текущего потока (номер строки результата)
    if (row < matrixSize) {
        float sum = 0.0f;
        for (int col = 0; col < vectorSize; col++) {
            float product = vector[col] * matrix[row * matrixSize + col]; // Произведение элемента вектора и элемента матрицы
            // printf("Multiplying vector[%d] * matrix[%d][%d]: %f * %f = %f\n", col, row, col, vector[col], matrix[row * matrixSize + col], product);
            sum += product; // Суммируем произведения
        }
        result[row] = sum; // Записываем результат в вектор
    }
}

int main() {
    // Указание размера матрицы и вектора
    int vectorSize = VECTOR_SIZE;
    int matrixSize = MATRIX_SIZE;

    // Выделение памяти для вектора, матрицы и результата на хосте
    float *h_vector = (float*)malloc(vectorSize * sizeof(float));
    float *h_matrix = (float*)malloc(matrixSize * matrixSize * sizeof(float));
    float *h_result = (float*)malloc(matrixSize * sizeof(float));

    // Инициализация вектора
    for (int i = 0; i < vectorSize; i++) {
        h_vector[i] = i + 1.0f; // Пример заполнения вектора (1.0, 2.0, 3.0)
    }
    
    // Инициализация матрицы
    for (int i = 0; i < matrixSize * matrixSize; i++) {
        h_matrix[i] = (i % MATRIX_SIZE) + 1.0f; // Пример заполнения матрицы
    }

    // Выделение памяти на устройстве
    float *d_vector, *d_matrix, *d_result;
    cudaMalloc((void**)&d_vector, vectorSize * sizeof(float));
    cudaMalloc((void**)&d_matrix, matrixSize * matrixSize * sizeof(float));
    cudaMalloc((void**)&d_result, matrixSize * sizeof(float));

    // Копирование данных из хоста на устройство
    cudaMemcpy(d_vector, h_vector, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix, h_matrix, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);

    // Определение размеров блоков и сетки
    int threadsPerBlock = 256; // Количество потоков на блок
    int blocksPerGrid = (matrixSize + threadsPerBlock - 1) / threadsPerBlock; // Количество блоков

    // Вызов ядра
    multiplyVectorByMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_vector, d_matrix, d_result, vectorSize, matrixSize);
    
    // Синхронизация потоков, чтобы убедиться, что все выводы были выполнены
    cudaDeviceSynchronize();

    // Копирование результата с устройства на хост
    cudaMemcpy(h_result, d_result, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Печать результата
    printf("Result of vector * matrix:\n");
    for (int i = 0; i < matrixSize; i++) {
        printf("%f\n", h_result[i]);
    }

    // Освобождение памяти
    free(h_vector);
    free(h_matrix);
    free(h_result);
    cudaFree(d_vector);
    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}