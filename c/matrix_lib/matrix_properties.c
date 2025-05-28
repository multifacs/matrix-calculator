#include <math.h>    
#include "matrix.h"  
#include "../constants.h"

// Helper function to check if a number is a power of two
int is_power_of_two(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

// Checks if the matrix is diagonal
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

// Checks if the matrix is symmetric
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

// Checks if the matrix is orthogonal
int is_orthogonal(matrix m) {
    if (m.rows != m.cols) return 0;
    matrix transp = transpose_matrix(m);
    matrix product = multiply_matrices(m, transp);
    matrix identity = create_identity_matrix(m.rows);
    int equal = matrices_equal(product, identity);

    free_matrix(&transp);
    free_matrix(&product);
    free_matrix(&identity);

    return equal;
}

// Checks if the matrix is upper triangular
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

// Checks if the matrix is lower triangular
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

// Checks if the matrix is an identity matrix
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