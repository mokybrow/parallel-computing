# parallel-computing
Параллельные и распределенные вычисления на CUDA

1. Сложение векторов - nvcc -o vector_addition vector_addition.cu
2. Умножение векторов - nvcc -o vector_multiply vector_multiply.cu
3. Умножение ветора на матрицу - nvcc -o vector_matrix_multiplication vector_matrix_multiplication.cu
4. Умножение матрицы на матрицу - nvcc -o matrix_multiplication matrix_multiplication.cu
5. Решение СЛАУ - nvcc -o matrix_solver matrix_solver.cu
6. Перемножение матриц с API cublas - nvcc -o matrix_solver_cublas matrix_solver_cublas.cu -lcublas
7. Решение СЛАУ с API cublas - nvcc -o matrix_multiplication_cublas matrix_multiplication_cublas.cu -lcublas
