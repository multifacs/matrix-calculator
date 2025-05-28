#include <stdio.h>   
#include <stdlib.h>  
#include <math.h>    
#include "matrix.h"  
#include "../constants.h"

/* 
 * Performs Cholesky decomposition on a symmetric positive definite matrix.
 * Efficient for symmetric positive definite matrices, reducing A to L*L^T.
 * Computes a lower triangular matrix L such that A = L*L^T, leveraging symmetry 
 * and positive definiteness for stability and efficiency.
 */
matrix cholesky_decomposition(matrix m) {
    if (m.rows != m.cols) {
        printf("%sError: matrix must be square for Cholesky decomposition.\n%s", URED, COLOR_RESET);
        exit(1);
    }
    if (!is_symmetric(m)) {
        printf("%sError: matrix must be symmetric for Cholesky decomposition\n%s", URED, COLOR_RESET);
        exit(1);
    }

    int n = m.rows;

    // Check positive definiteness by ensuring all leading minors have positive determinants
    for (int k = 1; k <= n; k++) {
        matrix minor = create_matrix(k, k);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                minor.data[i][j] = m.data[i][j];
            }
        }

        double det = determinant(minor);
        free_matrix(&minor);
        if (det <= 0) {
            printf("%sError: matrix is not positive definite.\n%s", URED, COLOR_RESET);
            exit(1);
        }
    }

    matrix L = create_matrix(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L.data[i][j] = 0.0;     // Initialize L as lower triangular matrix
        }
    }
    for (int k = 0; k < n; k++) {
        double sum1 = 0.0;
        for (int m = 0; m < k; m++) {
            sum1 += L.data[k][m] * L.data[k][m];    // Compute diagonal element contribution
        }
        double temp = m.data[k][k] - sum1;
        if (temp <= 0) {
            printf("%sError: matrix is not positive definite.\n%s", URED, COLOR_RESET);
            free_matrix(&L);
            exit(1);
        }
        L.data[k][k] = sqrt(temp);      // Diagonal element is sqrt of remaining value
        for (int i = k + 1; i < n; i++) {
            double sum2 = 0.0;
            for (int m = 0; m < k; m++) {
                sum2 += L.data[i][m] * L.data[k][m];    // Compute off-diagonal contribution
            }
            L.data[i][k] = (m.data[i][k] - sum2) / L.data[k][k];    // Compute L[i][k]
        }
    }
    return L;
}

// Helper function for Householder reflection to assist QR decomposition
// Constructs a reflection matrix to zero out elements below the diagonal
matrix householder_reflection(matrix a, int k) {
    int n = a.rows;
    if (k >= n - 1) {
        return create_identity_matrix(n);
    }
    double norm = 0.0;
    for (int i = k; i < n; i++) {
        norm += a.data[i][k] * a.data[i][k];
    }
    norm = sqrt(norm);
    if (norm < TOLERANCE) { 
        return create_identity_matrix(n);
    }
    double x_k = a.data[k][k];
    double sign = (x_k >= 0) ? 1.0 : -1.0;
    double v_k = x_k + sign * norm; // v[0] = x_k + sign(x_k) * norm
    double vTv = v_k * v_k;
    for (int i = k + 1; i < n; i++) {
        vTv += a.data[i][k] * a.data[i][k];
    }
    if (vTv < TOLERANCE) {
        return create_identity_matrix(n);
    }
    double beta = 2.0 / vTv;
    matrix Q = create_identity_matrix(n);
    for (int i = k; i < n; i++) {
        for (int j = k; j < n; j++) {
            double v_i = (i == k) ? v_k : a.data[i][k];
            double v_j = (j == k) ? v_k : a.data[j][k];
            Q.data[i][j] -= beta * v_i * v_j;
        }
    }
    return Q;
}

// Performs QR decomposition using Householder reflection
// Decompose matrix m into orthogonal Q and upper triangular R
void qr_decomposition(matrix m, matrix *Q, matrix *R) {
    int n = m.rows;
    *Q = create_identity_matrix(n);
    *R = create_matrix(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            (*R).data[i][j] = m.data[i][j];
        }
    }
    for (int k = 0; k < n - 1; k++) {
        matrix H = householder_reflection(*R, k);
        matrix temp = multiply_matrices(*Q, H);
        free_matrix(Q);
        *Q = temp;
        matrix temp2 = multiply_matrices(H, *R);
        free_matrix(R);
        *R = temp2;
        free_matrix(&H);
    }
}

/* 
 * Performs LU decomposition on the matrix.
 * Factorizes A into L*U.
 */
int lu_decomposition(matrix m, matrix *L, matrix *U) {
    if (m.rows != m.cols) {
        printf("%sError: matrix must be square for LU decomposition.\n%s", URED, COLOR_RESET);
        return 1;
    }

    int n = m.rows;
    *L = create_matrix(n, n);
    *U = create_matrix(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                (*L).data[i][j] = 1.0;      // L has 1s in diagonal
            } else {
                (*L).data[i][j] = 0.0;
            }
            (*U).data[i][j] = m.data[i][j];     // U starts as copy of m
        }
    }

    for (int k = 0; k < n; k++) {
        if (fabs((*U).data[k][k]) < TOLERANCE) { 
            free_matrix(L);
            free_matrix(U);
            return 1; 
        }
        for (int i = k + 1; i < n; i++) {
            double factor = (*U).data[i][k] / (*U).data[k][k];
            (*L).data[i][k] = factor;
            for (int j = k; j < n; j++) {
                (*U).data[i][j] -= factor * (*U).data[k][j];
            }
        }
    }
    return 0; // Successful decomposition
}

// Function for SVD computation with QR-algorithm
void svd(matrix A, matrix *U, matrix *Sigma, matrix *V) {
    int m = A.rows;
    int n = A.cols;

    matrix At = transpose_matrix(A);
    matrix AtA = multiply_matrices(At, A); // n x n

    matrix eigenvalues_V, eigenvectors_V;
    qr_algorithm(AtA, &eigenvalues_V, &eigenvectors_V, 2000, 1e-10);

    int min_dim = (m < n) ? m : n;
    double *singular_values = malloc(min_dim * sizeof(double));
    int *indices = malloc(min_dim * sizeof(int));
    for (int i = 0; i < min_dim; i++) {
        singular_values[i] = sqrt(fabs(eigenvalues_V.data[i][0]));
        indices[i] = i;
    }

    for (int i = 0; i < min_dim - 1; i++) {
        for (int j = 0; j < min_dim - i - 1; j++) {
            if (singular_values[j] < singular_values[j + 1]) {
                double temp = singular_values[j];
                singular_values[j] = singular_values[j + 1];
                singular_values[j + 1] = temp;
                int temp_idx = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp_idx;
            }
        }
    }

    *V = create_matrix(n, n);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            (*V).data[i][j] = eigenvectors_V.data[i][indices[j]];
        }
    }

    *Sigma = create_matrix(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            (*Sigma).data[i][j] = 0.0;
        }
    }
    for (int i = 0; i < min_dim; i++) {
        (*Sigma).data[i][i] = singular_values[i];
    }

    matrix Sigma_inv = create_matrix(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            Sigma_inv.data[i][j] = 0.0;
        }
    }
    for (int i = 0; i < min_dim; i++) {
        if (singular_values[i] > 1e-10) { // Защита от деления на ноль
            Sigma_inv.data[i][i] = 1.0 / singular_values[i];
        }
    }

    matrix temp = multiply_matrices(A, *V);
    *U = multiply_matrices(temp, Sigma_inv);

    for (int j = 0; j < (*U).cols; j++) {
        double norm = 0.0;
        for (int i = 0; i < (*U).rows; i++) {
            norm += (*U).data[i][j] * (*U).data[i][j];
        }
        norm = sqrt(norm);
        if (norm > 1e-10) { 
            for (int i = 0; i < (*U).rows; i++) {
                (*U).data[i][j] /= norm;
            }
        }
    }

    free(singular_values);
    free(indices);
    free_matrix(&At);
    free_matrix(&AtA);
    free_matrix(&eigenvalues_V);
    free_matrix(&eigenvectors_V);
    free_matrix(&temp);
    free_matrix(&Sigma_inv);
}

// Function for Schur decomposition A = QTQ^T
// Q orthogonal, T quasi-triangular
void schur_decomposition(matrix m, matrix *Q, matrix *T, int max_iter, double tol) {
    if (m.rows != m.cols) {
        printf("%sError: matrix must be square for Schur decomposition.\n%s", URED, COLOR_RESET);
        exit(1);
    }

    int n = m.rows;
    matrix A = create_matrix(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A.data[i][j] = m.data[i][j];
        }
    }

    *Q = create_identity_matrix(n);
    for (int iter = 0; iter < max_iter; iter++) {
        matrix Q_iter, R;
        qr_decomposition(A, &Q_iter, &R);
        matrix A_new = multiply_matrices(R, Q_iter);
        matrix Q_new = multiply_matrices(*Q, Q_iter);
        free_matrix(&A);
        A = A_new;
        free_matrix(Q);
        *Q = Q_new;
        free_matrix(&Q_iter);
        free_matrix(&R);

        int converged = 1;
        for (int i = 1; i < n; i++) {
            if (fabs(A.data[i][i - 1]) > tol) {
                converged = 0;
                break;
            }
        }
        if (converged) break;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs((*Q).data[i][j]) < DISPLAY_TOL) {
                (*Q).data[i][j] = 0.0;
            }
        }
    }

    *T = create_matrix(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i > j) {  
                (*T).data[i][j] = 0.0;
            } else {
                double val = A.data[i][j];
                if (fabs(val) < DISPLAY_TOL) {
                    val = 0.0;
                }
                (*T).data[i][j] = val;
            }
        }
    }

    free_matrix(&A);
}