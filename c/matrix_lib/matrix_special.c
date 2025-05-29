#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "../constants.h"

/**
 * @file matrix_special.c
 * @brief Functions for generating special types of matrices.
 *
 * This file contains functions to generate various special matrices such as Hilbert, Vandermonde,
 * Toeplitz, Hadamard, and Jacobi matrices. Each function creates a matrix of the specified type
 * and size, with error checking for invalid inputs or memory allocation failures.
 */

/**
 * @brief Generates a Hilbert matrix of size n x n.
 *
 * A Hilbert matrix is a square matrix with elements defined as 1 / (i + j + 1), where i and j
 * are the row and column indices starting from 0. Hilbert matrices are known for being ill-conditioned,
 * especially for larger values of n.
 *
 * @param n The size of the matrix (number of rows and columns).
 * @param m Pointer to the matrix where the Hilbert matrix will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if n <= 0,
 *         or INVALID_INPUT if memory allocation fails.
 */
int generate_hilbert_matrix(int n, matrix *m) {
    if (n <= 0) {
        return INVALID_DIMENSIONS;
    }
    *m = create_matrix(n, n);
    if (m->data == NULL) {
        return INVALID_INPUT;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m->data[i][j] = 1.0 / (i + j + 1);
        }
    }
    return SUCCESS;
}

/**
 * @brief Generates a Vandermonde matrix based on an array of values.
 *
 * A Vandermonde matrix is defined such that each row corresponds to the powers of a specific value.
 * Specifically, for a given array values of size n, the element at row i and column j is
 * values[i]^j, where i and j start from 0.
 *
 * @param n The size of the matrix (number of rows and columns).
 * @param values Array of `n` double values used to generate the matrix.
 * @param m Pointer to the matrix where the Vandermonde matrix will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if `n <= 0` or `values` is NULL,
 *         or INVALID_INPUT if memory allocation fails.
 */
int generate_vandermonde_matrix(int n, double* values, matrix *m) {
    if (n <= 0 || values == NULL) {
        return INVALID_DIMENSIONS;
    }
    *m = create_matrix(n, n);
    if (m->data == NULL) {
        return INVALID_INPUT;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m->data[i][j] = pow(values[i], j);
        }
    }
    return SUCCESS;
}

/**
 * @brief Generates a Toeplitz matrix from a given row.
 *
 * A Toeplitz matrix is a matrix in which each descending diagonal from left to right is constant.
 * This function generates a symmetric Toeplitz matrix using the provided row, where the first row
 * and first column are defined by the array `row`.
 *
 * @param n The size of the matrix (number of rows and columns).
 * @param row Array of `n` double values representing the first row of the matrix.
 * @param m Pointer to the matrix where the Toeplitz matrix will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if `n <= 0` or `row` is NULL,
 *         or INVALID_INPUT if memory allocation fails.
 */
int generate_toeplitz_matrix(int n, double* row, matrix *m) {
    if (n <= 0 || row == NULL) {
        return INVALID_DIMENSIONS;
    }
    *m = create_matrix(n, n);
    if (m->data == NULL) {
        return INVALID_INPUT;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j >= i) {
                m->data[i][j] = row[j - i];
            } else {
                m->data[i][j] = row[i - j];
            }
        }
    }
    return SUCCESS;
}

/**
 * @brief Generates a Hadamard matrix of size n x n.
 *
 * A Hadamard matrix is a square matrix with elements +1 or -1, and its rows are mutually orthogonal.
 * This function generates a Hadamard matrix using the Sylvester construction method, which requires
 * `n` to be a power of two.
 *
 * @param n The size of the matrix (must be a positive power of two).
 * @param m Pointer to the matrix where the Hadamard matrix will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_INPUT if `n` is not a positive power of two
 *         or if memory allocation fails.
 */
int generate_hadamard_matrix(int n, matrix *m) {
    if (n < 1 || !is_power_of_two(n)) {
        printf("%sError: n must be positive power of two.%s\n", URED, COLOR_RESET);
        return INVALID_INPUT;
    }

    *m = create_matrix(n, n);
    if (m->data == NULL) {
        return INVALID_INPUT;
    }
    m->data[0][0] = 1;

    // Use Sylvester's construction to build the Hadamard matrix
    for (int k = 1; k < n; k *= 2) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                m->data[i][j + k] = m->data[i][j];
                m->data[i + k][j] = m->data[i][j];
                m->data[i + k][j + k] = -m->data[i][j];
            }
        }
    }
    return SUCCESS;
}

/**
 * @brief Generates a Jacobi matrix with specified diagonal and off-diagonal values.
 *
 * A Jacobi matrix is a tridiagonal matrix with constant values `a` on the main diagonal and
 * constant values `b` on the subdiagonal and superdiagonal.
 *
 * @param n The size of the matrix (number of rows and columns).
 * @param a The value to place on the main diagonal.
 * @param b The value to place on the subdiagonal and superdiagonal.
 * @param m Pointer to the matrix where the Jacobi matrix will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if `n <= 0`,
 *         or INVALID_INPUT if memory allocation fails.
 */
int generate_jacobi_matrix(int n, double a, double b, matrix *m) {
    if (n <= 0) {
        return INVALID_DIMENSIONS;
    }
    *m = create_matrix(n, n);
    if (m->data == NULL) {
        return INVALID_INPUT;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                m->data[i][j] = a;
            } else if (i == j - 1 || i == j + 1) {
                m->data[i][j] = b;
            } else {
                m->data[i][j] = 0.0;
            }
        }
    }
    return SUCCESS;
}