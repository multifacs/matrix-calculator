#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>

// Matrix structure definition
typedef struct
{
    int rows;
    int cols;
    double **data;
} matrix;

// Create matrix of rows x cols size
matrix create_matrix(int rows, int cols);

// Free matrix memory
void free_matrix(matrix *m);

// Input elements of matrix
void input_matrix(matrix *m);

// Edit elements of matrix with validation
void edit_matrix(matrix *m);

// Print matrix
void print_matrix(matrix m);

// Function for matrix addition
matrix add_matrices(matrix a, matrix b);

// Fanction for matrix subtraction
matrix subtract_matrices(matrix a, matrix b);

// Function for matrix multiplication
matrix multiply_matrices(matrix a, matrix b);

// Function for matrix multiplication (using Strassen's algorithm)
matrix multiply_matrices_strassen(matrix a, matrix b);

// Helper function to check if a number is power of two
int is_power_of_two(int n);

// Helper function for combining matrices
matrix combine_matrix(matrix c11, matrix c12, matrix c21, matrix c22);

// Function for multiplying matrix by a scalar
matrix scalar_multiply(matrix m, double scalar);

// Function for matrix transposition
matrix transpose_matrix(matrix m);

// Function for finding minor of a matrix
matrix get_minor(matrix m, int row, int col);

// Evaluate matrix determinant
double determinant(matrix m);

// Helper function (Gaussian elimination)
void gaussian_elimination(matrix *m);

// Function for QR-algorithm to compute eigenvalues and eigenvectors
void qr_algorithm(matrix m, matrix *eigenvalues, matrix *eigenvectors, int max_iter, double tol);

// Function for finding inverse matrix (Gaussian-Jordan method)
matrix inverse_matrix(matrix m);

// Solve system of a linear equastions Ax = B
matrix solve_system(matrix A, matrix b);

// Function for finding matrix rank
int rank(matrix m);

// Function fro generation of a random matrix
matrix generate_random_matrix(int rows, int cols, double min_val, double max_val);

// Helper function for to create identity matrix
matrix create_identity_matrix(int size);

// Helper function to compare matrices
int matrices_equal(matrix a, matrix b);

// Function for finding power of a matrix
matrix matrix_power(matrix m, int exponent);

// Function for Cholesky decomposition (defined for symmetric definite matrix)
matrix cholesky_decomposition(matrix m);

// Function for eigenvalue and eigenvector
void power_method(matrix m, double *eigenvalue, matrix *eigenvector, int max_iter, double tol);

// Function for LU-decomposition
void lu_decomposition(matrix m, matrix *L, matrix *U);

// Function for Frobenius form
double frobenius_norm(matrix m);

// Function for one-norm
double one_norm(matrix m);

// Function for infinity-norm
double infinity_norm(matrix m);

// Functions to check matrix properties
int is_diagonal(matrix m);
int is_symmetric(matrix m);
int is_orthogonal(matrix m);
int is_upper_triangular(matrix m);
int is_lower_triangular(matrix m);
int is_identity(matrix m);

#endif // MATRIX_H