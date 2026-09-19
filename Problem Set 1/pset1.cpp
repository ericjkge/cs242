#include "pset1.hpp"

/**
 * ==========================================================
 * PART 3.1: Inner Product Matrix Multiplication
 * ==========================================================
 */

/**
 * @brief Perform inner-product-based matrix-matrix multiplication.
 *
 * Computes C = C + A x B
 * A, B, and C are dense matrices stored row-major.
 *
 * @param C the result matrix array (length M x N).
 * @param A the first input matrix array (length M x K).
 * @param B the second input matrix array (length K x N).
 * @param M the height of matrix A.
 * @param N the width of matrix B.
 * @param K the width of matrix A.
 */
void inner_product_mmm(float *C, float const *A, float const *B, int M, int N, int K)
{
    // mnk order
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < K; k++)
            {
                C[i * N + j] += A[i * K + k] * B[k * N + j]; // Matrices are flat lists (so no vanilla 2D indexing)
            }
        }
    }
}

/**
 * ==========================================================
 * PART 3.2: Outer Product Matrix Multiplication
 * ==========================================================
 */

/**
 * @brief Perform outer-product-based matrix-matrix multiplication.
 *
 * Computes C = C + A x B
 * A, B, and C are dense matrices stored row-major.
 *
 * @param C the result matrix array (length M x N).
 * @param A the first input matrix array (length M x K).
 * @param B the second input matrix array (length K x N).
 * @param M the height of matrix A.
 * @param N the width of matrix B.
 * @param K the width of matrix A.
 */
void outer_product_mmm(float *C, float const *A, float const *B, int M, int N, int K)
{
    // kmn order
    for (int k = 0; k < K; k++)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
