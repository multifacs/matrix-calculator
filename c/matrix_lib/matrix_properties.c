#include <math.h>    
#include "matrix.h"  
#include "../constants.h"

/**
 * @brief Checks if a given integer is a power of two.
 *
 * This function determines whether the input integer n is a power of two.
 * A number is a power of two if it is greater than zero and has exactly one bit set in its binary representation.
 *
 * @param n The integer to check.
 * @return int 1 if `n` is a power of two, 0 otherwise.
 */
int is_power_of_two(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

/**
 * @brief Checks if a matrix is diagonal.
 *
 * A matrix is diagonal if all its off-diagonal elements are zero (within a tolerance).
 *
 * @param m The matrix to check.
 * @return int 1 if the matrix is diagonal, 0 otherwise.
 */
int is_diagonal (matrix m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            if (i != j && fabs(m.data[i][j]) > TOLERANCE) {
                return 0;
            }
        }
    }
    return 1;
}

/**
 * @brief Checks if a matrix is symmetric.
 *
 * A matrix is symmetric if it is square and equal to its transpose.
 * This function checks if the matrix is square and if the elements
 * satisfy m[i][j] == m[j][i] within a tolerance.
 *
 * @param m The matrix to check.
 * @return int 1 if the matrix is symmetric, 0 otherwise.
 */
int is_symmetric(matrix m) {
    if (m.rows != m.cols) return 0;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < i; j++) {
            if (fabs(m.data[i][j] - m.data[j][i]) > TOLERANCE) {
                return 0;
            }
        }
    }
    return 1;
}

/**
 * @brief Checks if a matrix is orthogonal.
 *
 * A matrix is orthogonal if it is square and its transpose is its inverse,
 * i.e., m * m^T = I, where I is the identity matrix.
 *
 * @param m The matrix to check.
 * @return int 1 if the matrix is orthogonal, 0 otherwise.
 */
int is_orthogonal(matrix m) {
    if (m.rows != m.cols) return 0;
    matrix transp = transpose_matrix(m);
    matrix product;
    int status = multiply_matrices(m, transp, &product);
    if (status != SUCCESS) {
        free_matrix(&transp);
        return 0;
    }
    matrix identity = create_identity_matrix(m.rows);
    int equal = matrices_equal(product, identity);

    free_matrix(&transp);
    free_matrix(&product);
    free_matrix(&identity);

    return equal;
}

/**
 * @brief Checks if a matrix is upper triangular.
 *
 * A matrix is upper triangular if all elements below the main diagonal are zero (within a tolerance).
 *
 * @param m The matrix to check.
 * @return int 1 if the matrix is upper triangular, 0 otherwise.
 */
int is_upper_triangular(matrix m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < i; j++) {
            if (fabs(m.data[i][j]) > TOLERANCE) {
                return 0;
            }
        }
    }
    return 1;
}

/**
 * @brief Checks if a matrix is lower triangular.
 *
 * A matrix is lower triangular if all elements above the main diagonal are zero (within a tolerance).
 *
 * @param m The matrix to check.
 * @return int 1 if the matrix is lower triangular, 0 otherwise.
 */
int is_lower_triangular(matrix m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = i + 1; j < m.cols; j++) {
            if (fabs(m.data[i][j]) > TOLERANCE) {
                return 0;
            }
        }
    }
    return 1;
}

/**
 * @brief Checks if a matrix is an identity matrix.
 *
 * An identity matrix is a square matrix with ones on the main diagonal and zeros elsewhere.
 *
 * @param m The matrix to check.
 * @return int 1 if the matrix is an identity matrix, 0 otherwise.
 */
int is_identity(matrix m) {
    if (m.rows != m.cols) return 0;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            if (i == j && fabs(m.data[i][j] - 1.0) > TOLERANCE) {
                return 0;
            } else if (i != j && fabs(m.data[i][j]) > TOLERANCE) {
                return 0;
            }
        }
    }
    return 1;
}