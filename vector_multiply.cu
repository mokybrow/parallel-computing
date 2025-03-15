#include <stdio.h>
#include <cuda.h>

__global__ void vectorMultiply(float *A, float *B, float *C, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < N) {
        // Calculate the product and accumulate it in C[0] using atomicAdd
        float product = A[index] * B[index];
        printf("Intermediate product A[%d] * B[%d] = %f\n", index, index, product);
        atomicAdd(&C[0], product);
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

int main() {
    int N = 10; // Size of the vectors
    size_t size = N * sizeof(float);
    
    // Allocate memory on the host
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(sizeof(float)); // For the result

    // Initialize vectors A and B
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f + i; // Example values
        h_B[i] = 2.0f + i; // Example values
    }
    *h_C = 0.0f; // Initialize the result

    // Print vectors A and B
    printVector(h_A, N, "Vector A");
    printVector(h_B, N, "Vector B");

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (N + 255) / 256;
    vectorMultiply<<<blocks, 256>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("Sum of products of the vectors: %f\n", *h_C);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}