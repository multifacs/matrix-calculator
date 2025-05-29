#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "../constants.h"

/**
 * @brief Performs Cholesky decomposition on a symmetric positive-definite matrix.
 *
 * This function decomposes a symmetric positive-definite matrix m into a lower triangular matrix L
 * such that m = L * L^T. It checks if the matrix is square, symmetric, and positive-definite by
 * examining the determinants of all leading principal minors.
 *
 * @param m The input matrix (must be square, symmetric, and positive-definite).
 * @param L Pointer to a matrix where the lower triangular matrix will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if the matrix is not square,
 *         NOT_SYMMETRIC if the matrix is not symmetric, NOT_POSITIVE_DEFINITE if the matrix is not
 *         positive-definite, or INVALID_INPUT if memory allocation fails.
 */
int cholesky_decomposition(matrix m, matrix *L) {
    if (m.rows != m.cols) {
        printf("%sError: matrix must be square for Cholesky decomposition.\n%s", URED, COLOR_RESET);
        return INVALID_DIMENSIONS;
    }
    if (!is_symmetric(m)) {
        printf("%sError: matrix must be symmetric for Cholesky decomposition\n%s", URED, COLOR_RESET);
        return NOT_SYMMETRIC;
    }

    int n = m.rows;

    // Check if the matrix is positive-definite by verifying all leading principal minors
    for (int k = 1; k <= n; k++) {
        matrix minor = create_matrix(k, k);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                minor.data[i][j] = m.data[i][j];
            }
        }

        double det;
        int status = determinant(minor, &det);
        free_matrix(&minor);
        if (status != SUCCESS || det <= 0) {
            printf("%sError: matrix is not positive definite.\n%s", URED, COLOR_RESET);
            return NOT_POSITIVE_DEFINITE;
        }
    }

    // Initialize the lower triangular matrix L
    *L = create_matrix(n, n);
    if (L->data == NULL) {
        return INVALID_INPUT;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L->data[i][j] = 0.0;
        }
    }

    // Compute the Cholesky decomposition
    for (int k = 0; k < n; k++) {
        double sum1 = 0.0;
        for (int m = 0; m < k; m++) {
            sum1 += L->data[k][m] * L->data[k][m];
        }
        double temp = m.data[k][k] - sum1;
        if (temp <= 0) {
            printf("%sError: matrix is not positive definite.\n%s", URED, COLOR_RESET);
            free_matrix(L);
            return NOT_POSITIVE_DEFINITE;
        }
        L->data[k][k] = sqrt(temp);
        for (int i = k + 1; i < n; i++) {
            double sum2 = 0.0;
            for (int m = 0; m < k; m++) {
                sum2 += L->data[i][m] * L->data[k][m];
            }
            L->data[i][k] = (m.data[i][k] - sum2) / L->data[k][k];
        }
    }
    return SUCCESS;
}

/**
 * @brief Computes the Householder reflection matrix for a given column.
 *
 * This function generates a Householder reflection matrix that zeros out the subdiagonal elements
 * of the k-th column of matrix a below the k-th row. It is used as a step in QR decomposition.
 *
 * @param a The input matrix.
 * @param k The column index to apply the reflection to.
 * @return matrix The Householder reflection matrix (an identity matrix if no reflection is needed).
 */
matrix householder_reflection(matrix a, int k) {
    int n = a.rows;
    if (k >= n - 1) {
        return create_identity_matrix(n);
    }

    // Compute the norm of the subcolumn
    double norm = 0.0;
    for (int i = k; i < n; i++) {
        norm += a.data[i][k] * a.data[i][k];
    }
    norm = sqrt(norm);
    if (norm < TOLERANCE) { 
        return create_identity_matrix(n);
    }

    // Compute the Householder vector
    double x_k = a.data[k][k];
    double sign = (x_k >= 0) ? 1.0 : -1.0;
    double v_k = x_k + sign * norm;
    double vTv = v_k * v_k;
    for (int i = k + 1; i < n; i++) {
        vTv += a.data[i][k] * a.data[i][k];
    }
    if (vTv < TOLERANCE) {
        return create_identity_matrix(n);
    }

    // Compute the Householder matrix Q = I - beta * v * v^T
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

/**
 * @brief Performs QR decomposition on a square matrix.
 *
 * This function decomposes a square matrix m into an orthogonal matrix Q and an upper triangular
 * matrix R such that m = Q * R. It uses Householder reflections to achieve the decomposition.
 *
 * @param m The input matrix (must be square).
 * @param Q Pointer to a matrix where the orthogonal matrix will be stored.
 * @param R Pointer to a matrix where the upper triangular matrix will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_INPUT if memory allocation fails,
 *         or error codes from matrix operations.
 */
int qr_decomposition(matrix m, matrix *Q, matrix *R) {
    int n = m.rows;
    *Q = create_identity_matrix(n);
    *R = create_matrix(n, n);
    if (Q->data == NULL || R->data == NULL) {
        return INVALID_INPUT;
    }

    // Copy the input matrix to R
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            (*R).data[i][j] = m.data[i][j];
        }
    }

    // Apply Householder reflections
    for (int k = 0; k < n - 1; k++) {
        matrix H = householder_reflection(*R, k);
        matrix temp;
        int status = multiply_matrices(*Q, H, &temp);
        if (status != SUCCESS) {
            free_matrix(&H);
            free_matrix(Q);
            free_matrix(R);
            return status;
        }
        free_matrix(Q);
        *Q = temp;

        matrix temp2;
        status = multiply_matrices(H, *R, &temp2);
        if (status != SUCCESS) {
            free_matrix(&H);
            free_matrix(Q);
            free_matrix(R);
            return status;
        }
        free_matrix(R);
        *R = temp2;
        free_matrix(&H);
    }
    return SUCCESS;
}

/**
 * @brief Performs LU decomposition on a square matrix.
 *
 * This function decomposes a square matrix m into a lower triangular matrix L and an upper
 * triangular matrix U such that m = L * U. It uses Gaussian elimination without pivoting.
 *
 * @param m The input matrix (must be square).
 * @param L Pointer to a matrix where the lower triangular matrix will be stored.
 * @param U Pointer to a matrix where the upper triangular matrix will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if the matrix is not square,
 *         SINGULAR_MATRIX if the matrix is singular, or INVALID_INPUT if memory allocation fails.
 */
int lu_decomposition(matrix m, matrix *L, matrix *U) {
    if (m.rows != m.cols) {
        printf("%sError: matrix must be square for LU decomposition.\n%s", URED, COLOR_RESET);
        return INVALID_DIMENSIONS;
    }

    int n = m.rows;
    *L = create_matrix(n, n);
    *U = create_matrix(n, n);
    if (L->data == NULL || U->data == NULL) {
        free_matrix(L);
        free_matrix(U);
        return INVALID_INPUT;
    }

    // Initialize L as an identity matrix and U as a copy of m
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                (*L).data[i][j] = 1.0;
            } else {
                (*L).data[i][j] = 0.0;
            }
            (*U).data[i][j] = m.data[i][j];
        }
    }

    // Perform Gaussian elimination to compute L and U
    for (int k = 0; k < n; k++) {
        if (fabs((*U).data[k][k]) < TOLERANCE) { 
            free_matrix(L);
            free_matrix(U);
            return SINGULAR_MATRIX; 
        }
        for (int i = k + 1; i < n; i++) {
            double factor = (*U).data[i][k] / (*U).data[k][k];
            (*L).data[i][k] = factor;
            for (int j = k; j < n; j++) {
                (*U).data[i][j] -= factor * (*U).data[k][j];
            }
        }
    }
    return SUCCESS;
}

/**
 * @brief Performs Singular Value Decomposition (SVD) on a matrix.
 *
 * This function decomposes a matrix A into U * Sigma * V^T, where U and V are orthogonal
 * matrices, and Sigma is a diagonal matrix containing singular values. It uses the QR algorithm
 * to compute eigenvalues and eigenvectors of A^T * A.
 *
 * @param A The input matrix.
 * @param U Pointer to a matrix where the left singular vectors will be stored.
 * @param Sigma Pointer to a matrix where the singular values will be stored.
 * @param V Pointer to a matrix where the right singular vectors will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if the matrix dimensions are invalid,
 *         INVALID_INPUT if memory allocation fails, or error codes from matrix operations.
 */
int svd(matrix A, matrix *U, matrix *Sigma, matrix *V) {
    int m = A.rows;
    int n = A.cols;

    if (m <= 0 || n <= 0) {
        return INVALID_DIMENSIONS;
    }

    // Compute A^T * A
    matrix At = transpose_matrix(A);
    matrix AtA;
    int status = multiply_matrices(At, A, &AtA);
    if (status != SUCCESS) {
        free_matrix(&At);
        return status;
    }

    // Compute eigenvalues and eigenvectors of A^T * A
    matrix eigenvalues_V, eigenvectors_V;
    int error = qr_algorithm(AtA, &eigenvalues_V, &eigenvectors_V, 2000, 1e-10);
    if (error != SUCCESS) {
        free_matrix(&At);
        free_matrix(&AtA);
        return error;
    }

    // Compute singular values
    int min_dim = (m < n) ? m : n;
    double *singular_values = malloc(min_dim * sizeof(double));
    int *indices = malloc(min_dim * sizeof(int));
    if (!singular_values || !indices) {
        free_matrix(&At);
        free_matrix(&AtA);
        free_matrix(&eigenvalues_V);
        free_matrix(&eigenvectors_V);
        free(singular_values);
        free(indices);
        return INVALID_INPUT;
    }

    for (int i = 0; i < min_dim; i++) {
        singular_values[i] = sqrt(fabs(eigenvalues_V.data[i][0]));
        indices[i] = i;
    }

    // Sort singular values in descending order
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

    // Initialize V matrix
    *V = create_matrix(n, n);
    if (V->data == NULL) {
        free_matrix(&At);
        free_matrix(&AtA);
        free_matrix(&eigenvalues_V);
        free_matrix(&eigenvectors_V);
        free(singular_values);
        free(indices);
        return INVALID_INPUT;
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            (*V).data[i][j] = eigenvectors_V.data[i][indices[j]];
        }
    }

    // Initialize Sigma matrix
    *Sigma = create_matrix(m, n);
    if (Sigma->data == NULL) {
        free_matrix(&At);
        free_matrix(&AtA);
        free_matrix(&eigenvalues_V);
        free_matrix(&eigenvectors_V);
        free_matrix(V);
        free(singular_values);
        free(indices);
        return INVALID_INPUT;
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            (*Sigma).data[i][j] = 0.0;
        }
    }
    for (int i = 0; i < min_dim; i++) {
        (*Sigma).data[i][i] = singular_values[i];
    }

    // Compute pseudo-inverse of Sigma
    matrix Sigma_inv = create_matrix(n, m);
    if (Sigma_inv.data == NULL) {
        free_matrix(&At);
        free_matrix(&AtA);
        free_matrix(&eigenvalues_V);
        free_matrix(&eigenvectors_V);
        free_matrix(V);
        free_matrix(Sigma);
        free(singular_values);
        free(indices);
        return INVALID_INPUT;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            Sigma_inv.data[i][j] = 0.0;
        }
    }
    for (int i = 0; i < min_dim; i++) {
        if (singular_values[i] > 1e-10) {
            Sigma_inv.data[i][i] = 1.0 / singular_values[i];
        }
    }

    // Compute U matrix: U = A * V * Sigma_inv
    matrix temp;
    status = multiply_matrices(A, *V, &temp);
    if (status != SUCCESS) {
        free_matrix(&At);
        free_matrix(&AtA);
        free_matrix(&eigenvalues_V);
        free_matrix(&eigenvectors_V);
        free_matrix(V);
        free_matrix(Sigma);
        free_matrix(&Sigma_inv);
        free(singular_values);
        free(indices);
        return status;
    }
    matrix U_temp;
    status = multiply_matrices(temp, Sigma_inv, &U_temp);
    if (status != SUCCESS) {
        free_matrix(&temp);
        free_matrix(&At);
        free_matrix(&AtA);
        free_matrix(&eigenvalues_V);
        free_matrix(&eigenvectors_V);
        free_matrix(V);
        free_matrix(Sigma);
        free_matrix(&Sigma_inv);
        free(singular_values);
        free(indices);
        return status;
    }
    *U = U_temp;

    // Normalize columns of U
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

    // Clean up
    free(singular_values);
    free(indices);
    free_matrix(&At);
    free_matrix(&AtA);
    free_matrix(&eigenvalues_V);
    free_matrix(&eigenvectors_V);
    free_matrix(&temp);
    free_matrix(&Sigma_inv);
    return SUCCESS;
}

/**
 * @brief Performs Schur decomposition on a square matrix.
 *
 * This function decomposes a square matrix m into Q * T * Q^T, where Q is an orthogonal matrix
 * and T is an upper triangular matrix (or block upper triangular for complex eigenvalues).
 * It uses the QR algorithm with iterative QR decompositions.
 *
 * @param m The input matrix (must be square).
 * @param Q Pointer to a matrix where the orthogonal matrix will be stored.
 * @param T Pointer to a matrix where the upper triangular matrix will be stored.
 * @param max_iter The maximum number of iterations for the QR algorithm.
 * @param tol The tolerance for convergence.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if the matrix is not square,
 *         INVALID_INPUT if memory allocation fails, or error codes from matrix operations.
 */
int schur_decomposition(matrix m, matrix *Q, matrix *T, int max_iter, double tol) {
    if (m.rows != m.cols) {
        printf("%sError: matrix must be square for Schur decomposition.\n%s", URED, COLOR_RESET);
        return INVALID_DIMENSIONS;
    }

    int n = m.rows;
    matrix A = create_matrix(n, n);
    if (A.data == NULL) {
        return INVALID_INPUT;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A.data[i][j] = m.data[i][j];
        }
    }

    *Q = create_identity_matrix(n);
    if (Q->data == NULL) {
        free_matrix(&A);
        return INVALID_INPUT;
    }

    // Iterate using the QR algorithm
    for (int iter = 0; iter < max_iter; iter++) {
        matrix Q_iter, R;
        int status = qr_decomposition(A, &Q_iter, &R);
        if (status != SUCCESS) {
            free_matrix(&A);
            free_matrix(Q);
            return status;
        }
        matrix A_new;
        status = multiply_matrices(R, Q_iter, &A_new);
        if (status != SUCCESS) {
            free_matrix(&A);
            free_matrix(Q);
            free_matrix(&Q_iter);
            free_matrix(&R);
            return status;
        }
        matrix Q_new;
        status = multiply_matrices(*Q, Q_iter, &Q_new);
        if (status != SUCCESS) {
            free_matrix(&A);
            free_matrix(Q);
            free_matrix(&Q_iter);
            free_matrix(&R);
            free_matrix(&A_new);
            return status;
        }
        free_matrix(&A);
        A = A_new;
        free_matrix(Q);
        *Q = Q_new;
        free_matrix(&Q_iter);
        free_matrix(&R);

        // Check for convergence
        int converged = 1;
        for (int i = 1; i < n; i++) {
            if (fabs(A.data[i][i - 1]) > tol) {
                converged = 0;
                break;
            }
        }
        if (converged) break;
    }

    // Clean up small values in Q
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs((*Q).data[i][j]) < DISPLAY_TOL) {
                (*Q).data[i][j] = 0.0;
            }
        }
    }

    // Initialize T as an upper triangular matrix
    *T = create_matrix(n, n);
    if (T->data == NULL) {
        free_matrix(&A);
        free_matrix(Q);
        return INVALID_INPUT;
    }
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
    return SUCCESS;
}

/**
 * @brief Reduces a square matrix to Hessenberg form.
 *
 * This function transforms a square matrix A into a Hessenberg matrix H (where all entries below
 * the first subdiagonal are zero) using Householder reflections. It also computes an orthogonal matrix
 * Q such that H = Q^T * A * Q.
 *
 * @param A The input matrix (must be square).
 * @param H Pointer to a matrix where the Hessenberg matrix will be stored.
 * @param Q Pointer to a matrix where the orthogonal matrix will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if the matrix is not square,
 *         INVALID_INPUT if memory allocation fails, or error codes from matrix operations.
 */
int hessenberg_form(matrix A, matrix *H, matrix *Q) {
    if (A.rows != A.cols) {
        printf("%sError: matrix must be square for Hessenberg reduction.\n%s", URED, COLOR_RESET);
        return INVALID_DIMENSIONS;
    }

    int n = A.rows;
    *H = create_matrix(n, n);
    if (H->data == NULL) {
        return INVALID_INPUT;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            (*H).data[i][j] = A.data[i][j];
        }
    }

    *Q = create_identity_matrix(n);
    if (Q->data == NULL) {
        free_matrix(H);
        return INVALID_INPUT;
    }

    // Apply Householder reflections to reduce to Hessenberg form
    for (int k = 0; k < n - 2; k++) {
        matrix sub_col = create_matrix(n - k - 1, 1);
        if (sub_col.data == NULL) {
            free_matrix(H);
            free_matrix(Q);
            return INVALID_INPUT;
        }
        for (int i = 0; i < n - k - 1; i++) {
            sub_col.data[i][0] = (*H).data[k + 1 + i][k];
        }

        // Compute the norm of the subcolumn
        double norm = 0.0;
        for (int i = 0; i < sub_col.rows; i++) {
            norm += sub_col.data[i][0] * sub_col.data[i][0];
        }
        norm = sqrt(norm);

        if (norm < TOLERANCE) {
            free_matrix(&sub_col);
            continue;
        }

        // Compute the Householder vector
        double x1 = sub_col.data[0][0];
        double sign = (x1 >= 0) ? 1.0 : -1.0;
        double v1 = x1 + sign * norm;
        sub_col.data[0][0] = v1;

        // Normalize the Householder vector
        double v_norm = 0.0;
        for (int i = 0; i < sub_col.rows; i++) {
            v_norm += sub_col.data[i][0] * sub_col.data[i][0];
        }
        v_norm = sqrt(v_norm);
        for (int i = 0; i < sub_col.rows; i++) {
            sub_col.data[i][0] /= v_norm;
        }

        // Compute the Householder matrix Hk
        matrix Hk = create_identity_matrix(n);
        if (Hk.data == NULL) {
            free_matrix(H);
            free_matrix(Q);
            free_matrix(&sub_col);
            return INVALID_INPUT;
        }
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                Hk.data[i][j] -= 2.0 * sub_col.data[i - k - 1][0] * sub_col.data[j - k - 1][0];
            }
        }

        // Update H = Hk * H
        matrix temp;
        int status = multiply_matrices(Hk, *H, &temp);
        if (status != SUCCESS) {
            free_matrix(H);
            free_matrix(Q);
            free_matrix(&Hk);
            free_matrix(&sub_col);
            return status;
        }
        free_matrix(H);
        *H = temp;

        // Update Q = Q * Hk
        matrix Q_temp;
        status = multiply_matrices(*Q, Hk, &Q_temp);
        if (status != SUCCESS) {
            free_matrix(H);
            free_matrix(Q);
            free_matrix(&Hk);
            free_matrix(&sub_col);
            return status;
        }
        free_matrix(Q);
        *Q = Q_temp;

        free_matrix(&Hk);
        free_matrix(&sub_col);
    }
    return SUCCESS;
}