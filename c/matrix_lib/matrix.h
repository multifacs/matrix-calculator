#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>

/**
 * @file matrix.h
 * @brief Header file for matrix operations and manipulations.
 *
 * This file defines the matrix structure and declares functions for creating, manipulating, 
 * and performing computations on matrices. It includes basic operations (e.g., addition, multiplication), 
 * advanced decompositions (e.g., LU, SVD), and utility functions for matrix properties.
 */

/**
 * @struct matrix
 * @brief Structure representing a matrix.
 *
 * Stores the dimensions and elements of a matrix.
 */
typedef struct {
    int rows;      /**< Number of rows in the matrix. */
    int cols;      /**< Number of columns in the matrix. */
    double **data; /**< 2D array of doubles holding matrix elements. */
} matrix;

// Basic Matrix Operations

/**
 * @brief Creates a matrix with specified dimensions.
 *
 * Allocates memory for a matrix of size rows x cols and initializes its dimensions.
 *
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return matrix Newly created matrix.
 */
matrix create_matrix(int rows, int cols);

/**
 * @brief Frees the memory allocated for a matrix.
 *
 * Releases the dynamically allocated memory for the matrix data and resets its dimensions.
 *
 * @param m Pointer to the matrix to deallocate.
 */
void free_matrix(matrix *m);

/**
 * @brief Inputs matrix elements from the user.
 *
 * Prompts the user to enter values for each element of the matrix.
 *
 * @param m Pointer to the matrix to fill with user input.
 */
void input_matrix(matrix *m);

/**
 * @brief Inputs elements for a column vector.
 *
 * Likely intended for vector input; implementation should be verified for correctness.
 *
 * @param row Number of rows in the vector.
 */
void input_vector(int row);

/**
 * @brief Edits matrix elements with input validation.
 *
 * Allows interactive modification of matrix elements with checks for valid input.
 *
 * @param m Pointer to the matrix to edit.
 */
void edit_matrix(matrix *m);

/**
 * @brief Prints the matrix to the console.
 *
 * Displays the matrix in a formatted, readable layout.
 *
 * @param m Matrix to print.
 */
void print_matrix(matrix m);

// Matrix Arithmetic Operations

/**
 * @brief Adds two matrices element-wise.
 *
 * Computes the sum of two matrices and stores the result in a third matrix.
 *
 * @param a First matrix operand.
 * @param b Second matrix operand.
 * @param result Pointer to the matrix to store the sum.
 * @return int Status code: 0 for success, non-zero for failure (e.g., dimension mismatch).
 */
int add_matrices(matrix a, matrix b, matrix *result);

/**
 * @brief Subtracts one matrix from another element-wise.
 *
 * Computes the difference of two matrices and stores the result in a third matrix.
 *
 * @param a Matrix to subtract from.
 * @param b Matrix to subtract.
 * @param result Pointer to the matrix to store the difference.
 * @return int Status code: 0 for success, non-zero for failure (e.g., dimension mismatch).
 */
int subtract_matrices(matrix a, matrix b, matrix *result);

/**
 * @brief Multiplies two matrices.
 *
 * Performs standard matrix multiplication and stores the result.
 *
 * @param a First matrix operand.
 * @param b Second matrix operand.
 * @param result Pointer to the matrix to store the product.
 * @return int Status code: 0 for success, non-zero for failure (e.g., incompatible dimensions).
 */
int multiply_matrices(matrix a, matrix b, matrix *result);

/**
 * @brief Multiplies two matrices using Strassen's algorithm.
 *
 * Implements an efficient divide-and-conquer multiplication algorithm, optimal for large matrices.
 *
 * @param a First matrix operand.
 * @param b Second matrix operand.
 * @param result Pointer to the matrix to store the product.
 * @return int Status code: 0 for success, non-zero for failure (e.g., non-power-of-two dimensions).
 */
int multiply_matrices_strassen(matrix a, matrix b, matrix *result);

// Helper Functions

/**
 * @brief Checks if a number is a power of two.
 *
 * Useful for algorithms like Strassen's multiplication that require specific matrix sizes.
 *
 * @param n Number to check.
 * @return int 1 if n is a power of two, 0 otherwise.
 */
int is_power_of_two(int n);

/**
 * @brief Combines four submatrices into one matrix.
 *
 * Reconstructs a matrix from four quadrants, typically used in divide-and-conquer algorithms.
 *
 * @param c11 Top-left submatrix.
 * @param c12 Top-right submatrix.
 * @param c21 Bottom-left submatrix.
 * @param c22 Bottom-right submatrix.
 * @return matrix Combined matrix.
 */
matrix combine_matrix(matrix c11, matrix c12, matrix c21, matrix c22);

// Scalar and Transpose Operations

/**
 * @brief Multiplies a matrix by a scalar.
 *
 * Scales all elements of the matrix by the given scalar value.
 *
 * @param m Matrix to scale.
 * @param scalar Scaling factor.
 * @return matrix Resulting scaled matrix.
 */
matrix scalar_multiply(matrix m, double scalar);

/**
 * @brief Computes the transpose of a matrix.
 *
 * Creates a new matrix with rows and columns swapped.
 *
 * @param m Matrix to transpose.
 * @return matrix Transposed matrix.
 */
matrix transpose_matrix(matrix m);

// Determinant and Inverse

/**
 * @brief Extracts a minor matrix by excluding a row and column.
 *
 * Used in determinant and inverse computations.
 *
 * @param m Original matrix.
 * @param row Row to exclude.
 * @param col Column to exclude.
 * @return matrix Minor matrix.
 */
matrix get_minor(matrix m, int row, int col);

/**
 * @brief Computes the determinant of a square matrix.
 *
 * Calculates the determinant, typically using a decomposition method.
 *
 * @param m Matrix to evaluate.
 * @param det Pointer to store the computed determinant.
 * @return int Status code: 0 for success, non-zero for failure (e.g., non-square matrix).
 */
int determinant(matrix m, double *det);

/**
 * @brief Performs Gaussian elimination on a matrix.
 *
 * Modifies the matrix in-place to row echelon form, used in rank or inverse calculations.
 *
 * @param m Pointer to the matrix to transform.
 */
void gaussian_elimination(matrix *m);

/**
 * @brief Computes eigenvalues and eigenvectors using the QR algorithm.
 *
 * Iteratively applies QR decomposition to approximate eigenvalues and eigenvectors.
 *
 * @param m Input matrix.
 * @param eigenvalues Pointer to store the eigenvalues.
 * @param eigenvectors Pointer to store the eigenvectors.
 * @param max_iter Maximum number of iterations.
 * @param tol Convergence tolerance.
 * @return int Status code: 0 for success, non-zero for failure.
 */
int qr_algorithm(matrix m, matrix *eigenvalues, matrix *eigenvectors, int max_iter, double tol);

/**
 * @brief Performs QR decomposition on a matrix.
 *
 * Decomposes the matrix into an orthogonal matrix Q and an upper triangular matrix R.
 *
 * @param m Input matrix.
 * @param Q Pointer to store the orthogonal matrix.
 * @param R Pointer to store the upper triangular matrix.
 * @return int Status code: 0 for success, non-zero for failure.
 */
int qr_decomposition(matrix m, matrix *Q, matrix *R);

/**
 * @brief Computes the inverse of a square matrix.
 *
 * Uses the Gauss-Jordan method to find the inverse.
 *
 * @param m Matrix to invert.
 * @param inv Pointer to store the inverse matrix.
 * @return int Status code: 0 for success, non-zero for failure (e.g., singular matrix).
 */
int inverse_matrix(matrix m, matrix *inv);

// Linear System Solving

/**
 * @brief Solves a system of linear equations Ax = b.
 *
 * Uses an appropriate method (e.g., LU decomposition) to find the solution vector x.
 *
 * @param A Coefficient matrix.
 * @param b Right-hand side vector.
 * @param x Pointer to store the solution vector.
 * @return int Status code: 0 for success, non-zero for failure (e.g., singular matrix).
 */
int solve_system(matrix A, matrix b, matrix *x);

// Matrix Properties and Utilities

/**
 * @brief Computes the rank of a matrix.
 *
 * Determines the number of linearly independent rows or columns using Gaussian elimination.
 *
 * @param m Matrix to evaluate.
 * @return int Rank of the matrix, or -1 on error.
 */
int rank(matrix m);

/**
 * @brief Generates a random matrix with elements in a given range.
 *
 * Creates a matrix with random values for testing or simulation purposes.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param min_val Minimum value for random elements.
 * @param max_val Maximum value for random elements.
 * @return matrix Generated random matrix.
 */
matrix generate_random_matrix(int rows, int cols, double min_val, double max_val);

/**
 * @brief Creates an identity matrix of specified size.
 *
 * Generates a square matrix with ones on the diagonal and zeros elsewhere.
 *
 * @param size Number of rows and columns.
 * @return matrix Identity matrix.
 */
matrix create_identity_matrix(int size);

/**
 * @brief Compares two matrices for equality within a tolerance.
 *
 * Checks if two matrices are equal, allowing for small numerical differences.
 *
 * @param a First matrix.
 * @param b Second matrix.
 * @return int 1 if equal within tolerance, 0 otherwise.
 */
int matrices_equal(matrix a, matrix b);

// Advanced Matrix Operations

/**
 * @brief Raises a matrix to an integer power.
 *
 * Computes the matrix power, handling positive and negative exponents (via inverse).
 *
 * @param m Matrix to exponentiate.
 * @param exponent Integer power.
 * @param result Pointer to store the resulting matrix.
 * @return int Status code: 0 for success, non-zero for failure (e.g., singular matrix for negative exponent).
 */
int matrix_power(matrix m, int exponent, matrix *result);

// Matrix Decompositions

/**
 * @brief Performs Cholesky decomposition on a symmetric positive-definite matrix.
 *
 * Decomposes the matrix into L * L^T, where L is lower triangular.
 *
 * @param m Symmetric positive-definite matrix.
 * @param L Pointer to store the lower triangular matrix.
 * @return int Status code: 0 for success, non-zero for failure (e.g., non-positive-definite matrix).
 */
int cholesky_decomposition(matrix m, matrix *L);

/**
 * @brief Performs LU decomposition on a square matrix.
 *
 * Decomposes the matrix into L * U, where L is lower triangular and U is upper triangular.
 *
 * @param m Input matrix.
 * @param L Pointer to store the lower triangular matrix.
 * @param U Pointer to store the upper triangular matrix.
 * @return int Status code: 0 for success, non-zero for failure (e.g., non-square matrix).
 */
int lu_decomposition(matrix m, matrix *L, matrix *U);

/**
 * @brief Performs Schur decomposition on a square matrix.
 *
 * Decomposes the matrix into Q * T * Q^T, where Q is orthogonal and T is upper triangular.
 *
 * @param m Input matrix.
 * @param Q Pointer to store the orthogonal matrix.
 * @param T Pointer to store the upper triangular matrix.
 * @param max_iter Maximum iterations for convergence.
 * @param tol Convergence tolerance.
 * @return int Status code: 0 for success, non-zero for failure.
 */
int schur_decomposition(matrix m, matrix *Q, matrix *T, int max_iter, double tol);

// Matrix Norms

/**
 * @brief Computes the Frobenius norm of a matrix.
 *
 * Calculates the square root of the sum of squared elements.
 *
 * @param m Matrix to evaluate.
 * @return double Frobenius norm value.
 */
double frobenius_norm(matrix m);

/**
 * @brief Computes the one-norm of a matrix.
 *
 * Calculates the maximum absolute column sum.
 *
 * @param m Matrix to evaluate.
 * @return double One-norm value.
 */
double one_norm(matrix m);

/**
 * @brief Computes the infinity-norm of a matrix.
 *
 * Calculates the maximum absolute row sum.
 *
 * @param m Matrix to evaluate.
 * @return double Infinity-norm value.
 */
double infinity_norm(matrix m);

// Singular Value Decomposition

/**
 * @brief Performs Singular Value Decomposition (SVD) on a matrix.
 *
 * Decomposes A into U * Sigma * V^T, where U and V are orthogonal, and Sigma is diagonal.
 *
 * @param A Input matrix.
 * @param U Pointer to store the left singular vectors.
 * @param Sigma Pointer to store the singular values (diagonal matrix).
 * @param V Pointer to store the right singular vectors.
 * @return int Status code: 0 for success, non-zero for failure.
 */
int svd(matrix A, matrix *U, matrix *Sigma, matrix *V);

// Strassen's Algorithm Helpers

/**
 * @brief Finds the next power of two greater than or equal to a number.
 *
 * Used for padding matrices in Strassen's algorithm.
 *
 * @param n Input number.
 * @return int Next power of two.
 */
int next_power_of_two(int n);

/**
 * @brief Pads a matrix with zeros to specified dimensions.
 *
 * Expands the matrix by adding zeros, preserving the original elements in the top-left corner.
 *
 * @param m Original matrix.
 * @param new_rows Target number of rows.
 * @param new_cols Target number of columns.
 * @return matrix Padded matrix.
 */
matrix pad_matrix(matrix m, int new_rows, int new_cols);

/**
 * @brief Multiplies matrices using Strassen's algorithm with padding.
 *
 * Extends Strassen's algorithm to handle non-square or non-power-of-two matrices by padding.
 *
 * @param a First matrix operand.
 * @param b Second matrix operand.
 * @param result Pointer to store the product.
 * @return int Status code: 0 for success, non-zero for failure.
 */
int multiply_matrices_strassen_padded(matrix a, matrix b, matrix *result);

// Matrix Property Checks

/**
 * @brief Checks if a matrix is diagonal.
 *
 * Verifies if all off-diagonal elements are zero.
 *
 * @param m Matrix to check.
 * @return int 1 if diagonal, 0 otherwise.
 */
int is_diagonal(matrix m);

/**
 * @brief Checks if a matrix is symmetric.
 *
 * Verifies if the matrix equals its transpose.
 *
 * @param m Matrix to check.
 * @return int 1 if symmetric, 0 otherwise.
 */
int is_symmetric(matrix m);

/**
 * @brief Checks if a matrix is orthogonal.
 *
 * Verifies if m * m^T equals the identity matrix.
 *
 * @param m Matrix to check.
 * @return int 1 if orthogonal, 0 otherwise.
 */
int is_orthogonal(matrix m);

/**
 * @brief Checks if a matrix is upper triangular.
 *
 * Verifies if all elements below the main diagonal are zero.
 *
 * @param m Matrix to check.
 * @return int 1 if upper triangular, 0 otherwise.
 */
int is_upper_triangular(matrix m);

/**
 * @brief Checks if a matrix is lower triangular.
 *
 * Verifies if all elements above the main diagonal are zero.
 *
 * @param m Matrix to check.
 * @return int 1 if lower triangular, 0 otherwise.
 */
int is_lower_triangular(matrix m);

/**
 * @brief Checks if a matrix is an identity matrix.
 *
 * Verifies if the matrix has ones on the diagonal and zeros elsewhere.
 *
 * @param m Matrix to check.
 * @return int 1 if identity, 0 otherwise.
 */
int is_identity(matrix m);

// Special Matrix Generation

/**
 * @brief Generates a Hilbert matrix of size n x n.
 *
 * Creates a matrix with elements 1 / (i + j - 1).
 *
 * @param n Size of the square matrix.
 * @param m Pointer to store the Hilbert matrix.
 * @return int Status code: 0 for success, non-zero for failure.
 */
int generate_hilbert_matrix(int n, matrix *m);

/**
 * @brief Generates a Vandermonde matrix from an array of values.
 *
 * Creates a matrix where each row contains powers of a given value.
 *
 * @param n Size of the square matrix.
 * @param values Array of base values.
 * @param m Pointer to store the Vandermonde matrix.
 * @return int Status code: 0 for success, non-zero for failure.
 */
int generate_vandermonde_matrix(int n, double* values, matrix *m);

/**
 * @brief Generates a Toeplitz matrix from a given row.
 *
 * Creates a matrix with constant diagonals based on the first row.
 *
 * @param n Size of the square matrix.
 * @param row Array representing the first row.
 * @param m Pointer to store the Toeplitz matrix.
 * @return int Status code: 0 for success, non-zero for failure.
 */
int generate_toeplitz_matrix(int n, double* row, matrix *m);

/**
 * @brief Generates a Hadamard matrix of size n x n.
 *
 * Creates a matrix with elements ±1, where rows are orthogonal (n must be a power of two).
 *
 * @param n Size of the square matrix (power of two).
 * @param m Pointer to store the Hadamard matrix.
 * @return int Status code: 0 for success, non-zero for failure.
 */
int generate_hadamard_matrix(int n, matrix *m);

/**
 * @brief Generates a Jacobi matrix with specified values.
 *
 * Creates a tridiagonal matrix with constant diagonal and off-diagonal elements.
 *
 * @param n Size of the square matrix.
 * @param a Value for the main diagonal.
 * @param b Value for the off-diagonals.
 * @param m Pointer to store the Jacobi matrix.
 * @return int Status code: 0 for success, non-zero for failure.
 */
int generate_jacobi_matrix(int n, double a, double b, matrix *m);

// Additional Decomposition

/**
 * @brief Reduces a matrix to Hessenberg form.
 *
 * Transforms the matrix into an upper Hessenberg matrix using orthogonal similarity.
 *
 * @param A Input matrix.
 * @param H Pointer to store the Hessenberg matrix.
 * @param Q Pointer to store the orthogonal transformation matrix.
 * @return int Status code: 0 for success, non-zero for failure.
 */
int hessenberg_form(matrix A, matrix *H, matrix *Q);

// Condition Number

/**
 * @brief Computes the condition number of a matrix.
 *
 * Calculates the condition number using a specified norm function (e.g., one_norm).
 *
 * @param m Matrix to evaluate.
 * @param norm_func Pointer to the norm function to use.
 * @return double Condition number, or -1.0 on error (e.g., singular matrix).
 */
double condition_number(matrix m, double (*norm_func)(matrix));

// SVD-based System Solver

/**
 * @brief Solves a linear system using Singular Value Decomposition.
 *
 * Solves Ax = b using SVD, suitable for ill-conditioned or non-square systems.
 *
 * @param A Coefficient matrix.
 * @param b Right-hand side vector.
 * @param x Pointer to store the solution vector.
 * @return int Status code: 0 for success, non-zero for failure.
 */
int solve_system_svd(matrix A, matrix b, matrix *x);

#endif // MATRIX_H