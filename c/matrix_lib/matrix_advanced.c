#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "../constants.h"

/**
 * @brief Calculates the determinant of a square matrix.
 *
 * This function computes the determinant of a given square matrix using LU decomposition.
 * If the matrix is not square, an error is printed, and the function returns an error code.
 *
 * @param m The input matrix (must be square).
 * @param det Pointer to a double where the determinant will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if the matrix is not square,
 *         or other error codes from lu_decomposition.
 */
int determinant(matrix m, double *det) {
    if (m.rows != m.cols) {
        printf("%sError: determinant is defined only for square matrices.\n%s", URED, COLOR_RESET);
        return INVALID_DIMENSIONS;
    }
    
    matrix L, U;
    int status = lu_decomposition(m, &L, &U);
    if (status == SINGULAR_MATRIX) {
        *det = 0.0;
        free_matrix(&L);
        free_matrix(&U);
        return SUCCESS;
    }
    if (status != SUCCESS) {
        free_matrix(&L);
        free_matrix(&U);
        return status;
    }

    *det = 1.0;
    for (int i = 0; i < m.rows; i++) {
        *det *= U.data[i][i];
    }

    free_matrix(&L);
    free_matrix(&U);
    return SUCCESS;
}

/**
 * @brief Computes the inverse of a square matrix.
 *
 * This function calculates the inverse of a given square matrix using LU decomposition.
 * If the matrix is not square or is singular, an error is printed, and the function returns an error code.
 *
 * @param m The input matrix (must be square and non-singular).
 * @param inv Pointer to a matrix where the inverse will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if the matrix is not square,
 *         SINGULAR_MATRIX if the matrix is singular, or other error codes.
 */
int inverse_matrix(matrix m, matrix *inv) {
    if (m.rows != m.cols) {
        printf("%sError: inverse is only defined for square matrices.\n%s", URED, COLOR_RESET);
        return INVALID_DIMENSIONS;
    }

    matrix L, U;
    int status = lu_decomposition(m, &L, &U);
    if (status == SINGULAR_MATRIX) {
        printf("%sError: matrix is singular and cannot be inverted.\n%s", URED, COLOR_RESET);
        free_matrix(&L);
        free_matrix(&U);
        return SINGULAR_MATRIX;
    }
    if (status != SUCCESS) {
        free_matrix(&L);
        free_matrix(&U);
        return status;
    }

    int n = m.rows;
    *inv = create_matrix(n, n);
    if (inv->data == NULL) {
        free_matrix(&L);
        free_matrix(&U);
        return INVALID_INPUT;
    }

    for (int col = 0; col < n; col++) {
        matrix e = create_matrix(n, 1);
        if (e.data == NULL) {
            free_matrix(&L);
            free_matrix(&U);
            free_matrix(inv);
            return INVALID_INPUT;
        }
        for (int i = 0; i < n; i++) {
            e.data[i][0] = (i == col) ? 1.0 : 0.0;
        }

        matrix y = create_matrix(n, 1);
        if (y.data == NULL) {
            free_matrix(&L);
            free_matrix(&U);
            free_matrix(inv);
            free_matrix(&e);
            return INVALID_INPUT;
        }
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L.data[i][j] * y.data[j][0];
            }
            y.data[i][0] = (e.data[i][0] - sum) / L.data[i][i];
        }

        matrix x = create_matrix(n, 1);
        if (x.data == NULL) {
            free_matrix(&L);
            free_matrix(&U);
            free_matrix(inv);
            free_matrix(&e);
            free_matrix(&y);
            return INVALID_INPUT;
        }
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += U.data[i][j] * x.data[j][0];
            }
            x.data[i][0] = (y.data[i][0] - sum) / U.data[i][i];
        }

        for (int i = 0; i < n; i++) {
            inv->data[i][col] = x.data[i][0];
        }

        free_matrix(&e);
        free_matrix(&y);
        free_matrix(&x);
    }

    free_matrix(&L);
    free_matrix(&U);
    return SUCCESS;
}

/**
 * @brief Solves a system of linear equations.
 *
 * This function solves the system A * x = b for x. It uses LU decomposition for square matrices
 * and SVD for non-square matrices or when the condition number is high.
 *
 * @param A The coefficient matrix.
 * @param b The right-hand side vector.
 * @param x Pointer to a matrix where the solution vector will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if dimensions are invalid,
 *         SINGULAR_MATRIX if the matrix is singular, or other error codes.
 */
int solve_system(matrix A, matrix b, matrix *x) {
    int m = A.rows;
    int n = A.cols;

    if (m <= 0 || n <= 0 || b.rows <= 0 || b.cols <= 0) {
        return INVALID_DIMENSIONS;
    }

    if (m == n) {
        double cond = condition_number(A, one_norm);
        if (cond > 1e6 || cond < 0) {
            return solve_system_svd(A, b, x);
        } else {
            matrix L, U;
            int status = lu_decomposition(A, &L, &U);
            if (status == SINGULAR_MATRIX) {
                printf("%sError: matrix is singular and cannot solve the system.\n%s", URED, COLOR_RESET);
                free_matrix(&L);
                free_matrix(&U);
                return SINGULAR_MATRIX;
            }
            if (status != SUCCESS) {
                free_matrix(&L);
                free_matrix(&U);
                return status;
            }

            *x = create_matrix(m, 1);
            if (x->data == NULL) {
                free_matrix(&L);
                free_matrix(&U);
                return INVALID_INPUT;
            }
            matrix y = create_matrix(m, 1);
            if (y.data == NULL) {
                free_matrix(&L);
                free_matrix(&U);
                free_matrix(x);
                return INVALID_INPUT;
            }
            for (int i = 0; i < m; i++) {
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    sum += L.data[i][j] * y.data[j][0];
                }
                y.data[i][0] = (b.data[i][0] - sum) / L.data[i][i];
            }

            for (int i = n - 1; i >= 0; i--) {
                double sum = 0.0;
                for (int j = i + 1; j < n; j++) {
                    sum += U.data[i][j] * x->data[j][0];
                }
                x->data[i][0] = (y.data[i][0] - sum) / U.data[i][i];
            }

            free_matrix(&L);
            free_matrix(&U);
            free_matrix(&y);
            return SUCCESS;
        }
    } else {
        return solve_system_svd(A, b, x);
    }
}

/**
 * @brief Solves a system of linear equations using SVD.
 *
 * This function solves the system A * x = b using Singular Value Decomposition (SVD).
 * Particularly useful for overdetermined or underdetermined systems.
 *
 * @param A The coefficient matrix.
 * @param b The right-hand side vector.
 * @param x Pointer to a matrix where the solution vector will be stored.
 * @return int Status code: SUCCESS if successful, or error codes from SVD or matrix operations.
 */
int solve_system_svd(matrix A, matrix b, matrix *x) {
    matrix U, Sigma, V;
    int status = svd(A, &U, &Sigma, &V);
    if (status != SUCCESS) return status;

    int m = A.rows;
    int n = A.cols;
    int min_dim = (m < n) ? m : n;

    matrix Sigma_pseudo = create_matrix(n, m);
    for (int i = 0; i < min_dim; i++) {
        if (fabs(Sigma.data[i][i]) > TOLERANCE) {
            Sigma_pseudo.data[i][i] = 1.0 / Sigma.data[i][i];
        }
    }

    matrix Ut_b;
    matrix U_t = transpose_matrix(U);
    status = multiply_matrices(U_t, b, &Ut_b);
    free_matrix(&U_t);
    if (status != SUCCESS) {
        free_matrix(&U);
        free_matrix(&Sigma);
        free_matrix(&V);
        free_matrix(&Sigma_pseudo);
        return status;
    }

    matrix Sigma_pseudo_Ut_b;
    status = multiply_matrices(Sigma_pseudo, Ut_b, &Sigma_pseudo_Ut_b);
    free_matrix(&Ut_b);
    free_matrix(&Sigma_pseudo);
    if (status != SUCCESS) {
        free_matrix(&U);
        free_matrix(&Sigma);
        free_matrix(&V);
        return status;
    }

    status = multiply_matrices(V, Sigma_pseudo_Ut_b, x);
    free_matrix(&Sigma_pseudo_Ut_b);
    free_matrix(&U);
    free_matrix(&Sigma);
    free_matrix(&V);
    return status;
}

/**
 * @brief Determines the rank of a matrix.
 *
 * This function calculates the rank of a matrix by performing Gaussian elimination and counting the number of non-zero rows.
 *
 * @param m The input matrix.
 * @return int The rank of the matrix, or -1 if there is an error.
 */
int rank(matrix m) {
    matrix temp = create_matrix(m.rows, m.cols);
    if (temp.data == NULL) {
        return -1;
    }
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            temp.data[i][j] = m.data[i][j];
        }
    }
    gaussian_elimination(&temp);
    int rank_count = 0;
    for (int i = 0; i < temp.rows; i++) {
        int zero_row = 1;
        for (int j = 0; j < temp.cols; j++) {
            if (fabs(temp.data[i][j]) > TOLERANCE) {
                zero_row = 0;
                break;
            }
        }
        if (!zero_row) rank_count++;
    }
    free_matrix(&temp);
    return rank_count;
}

/**
 * @brief Computes the matrix raised to an integer power.
 *
 * This function calculates m^exponent for a square matrix m. It supports both positive and negative exponents.
 * For negative exponents, it computes the inverse of the matrix raised to the absolute value of the exponent.
 *
 * @param m The input matrix (must be square).
 * @param exponent The integer exponent.
 * @param result Pointer to a matrix where the result will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if the matrix is not square,
 *         SINGULAR_MATRIX if the matrix is singular (for negative exponents), or other error codes.
 */
int matrix_power(matrix m, int exponent, matrix *result) {
    if (m.rows != m.cols) {
        printf("%sError: matrix must be square.%s\n", URED, COLOR_RESET);
        return INVALID_DIMENSIONS;
    }

    if (exponent == 0) {
        *result = create_identity_matrix(m.rows);
        if (result->data == NULL) {
            return INVALID_INPUT;
        }
        return SUCCESS;
    }

    *result = create_identity_matrix(m.rows);
    if (result->data == NULL) {
        return INVALID_INPUT;
    }
    matrix base = create_matrix(m.rows, m.cols);
    if (base.data == NULL) {
        free_matrix(result);
        return INVALID_INPUT;
    }
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            base.data[i][j] = m.data[i][j];
        }
    }

    int abs_exp = abs(exponent);
    while (abs_exp > 0) {
        if (abs_exp % 2 == 1) {
            matrix temp;
            int status = multiply_matrices(*result, base, &temp);
            if (status != SUCCESS) {
                free_matrix(result);
                free_matrix(&base);
                return status;
            }
            free_matrix(result);
            *result = temp;
        }

        matrix temp;
        int status = multiply_matrices(base, base, &temp);
        if (status != SUCCESS) {
            free_matrix(result);
            free_matrix(&base);
            return status;
        }
        free_matrix(&base);
        base = temp;
        abs_exp /= 2;
    }

    if (exponent < 0) {
        double det;
        int status = determinant(*result, &det);
        if (status != SUCCESS || fabs(det) < TOLERANCE) {
            printf("%sError: matrix is singular and cannot be inverted.%s\n", URED, COLOR_RESET);
            free_matrix(result);
            free_matrix(&base);
            return SINGULAR_MATRIX;
        }
        matrix inv;
        status = inverse_matrix(*result, &inv);
        if (status != SUCCESS) {
            free_matrix(result);
            free_matrix(&base);
            return status;
        }
        free_matrix(result);
        *result = inv;
    }

    free_matrix(&base);
    return SUCCESS;
}

/**
 * @brief Implements the QR algorithm for finding eigenvalues and eigenvectors.
 *
 * This function uses the QR algorithm to compute the eigenvalues and eigenvectors of a square matrix.
 * It iteratively applies QR decomposition until convergence or the maximum number of iterations is reached.
 *
 * @param m The input matrix (must be square).
 * @param eigenvalues Pointer to a matrix where the eigenvalues will be stored.
 * @param eigenvectors Pointer to a matrix where the eigenvectors will be stored.
 * @param max_iter The maximum number of iterations.
 * @param tol The tolerance for convergence.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if the matrix is not square,
 *         or error codes from QR decomposition or matrix operations.
 */
int qr_algorithm(matrix m, matrix *eigenvalues, matrix *eigenvectors, int max_iter, double tol) {
    if (m.rows != m.cols) {
        printf("%sError: matrix must be square.\n%s", URED, COLOR_RESET);
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
    matrix Q_total = create_identity_matrix(n);
    if (Q_total.data == NULL) {
        free_matrix(&A);
        return INVALID_INPUT;
    }
    for (int iter = 0; iter < max_iter; iter++) {
        matrix Q, R;
        int status = qr_decomposition(A, &Q, &R);
        if (status != SUCCESS) {
            free_matrix(&A);
            free_matrix(&Q_total);
            return status;
        }
        matrix A_new;
        status = multiply_matrices(R, Q, &A_new);
        if (status != SUCCESS) {
            free_matrix(&A);
            free_matrix(&Q_total);
            free_matrix(&Q);
            free_matrix(&R);
            return status;
        }
        matrix Q_total_new;
        status = multiply_matrices(Q_total, Q, &Q_total_new);
        if (status != SUCCESS) {
            free_matrix(&A);
            free_matrix(&Q_total);
            free_matrix(&Q);
            free_matrix(&R);
            free_matrix(&A_new);
            return status;
        }
        free_matrix(&A);
        A = A_new;
        free_matrix(&Q_total);
        Q_total = Q_total_new;
        free_matrix(&Q);
        free_matrix(&R);
        int converged = 1;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (fabs(A.data[i][j]) > tol) {
                    converged = 0;
                    break;
                }
            }
            if (!converged) break;
        }
        if (converged) break;
    }
    *eigenvalues = create_matrix(n, 1);
    if (eigenvalues->data == NULL) {
        free_matrix(&A);
        free_matrix(&Q_total);
        return INVALID_INPUT;
    }
    for (int i = 0; i < n; i++) {
        (*eigenvalues).data[i][0] = A.data[i][i];
    }
    *eigenvectors = Q_total;
    free_matrix(&A);
    return SUCCESS;
}

/**
 * @brief Calculates the Frobenius norm of a matrix.
 *
 * The Frobenius norm is the square root of the sum of the squares of all elements in the matrix.
 *
 * @param m The input matrix.
 * @return double The Frobenius norm of the matrix.
 */
double frobenius_norm(matrix m) {
    double sum = 0.0;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            sum += m.data[i][j] * m.data[i][j];
        }
    }
    return sqrt(sum);
}

/**
 * @brief Computes the one norm (maximum column sum) of a matrix.
 *
 * The one norm is the maximum absolute column sum of the matrix.
 *
 * @param m The input matrix.
 * @return double The one norm of the matrix.
 */
double one_norm(matrix m) {
    double max_sum = 0.0;
    for (int j = 0; j < m.cols; j++) {
        double col_sum = 0.0;
        for (int i = 0; i < m.rows; i++) {
            col_sum += fabs(m.data[i][j]);
        }
        if (col_sum > max_sum) {
            max_sum = col_sum;
        }
    }
    return max_sum;
}

/**
 * @brief Computes the infinity norm (maximum row sum) of a matrix.
 *
 * The infinity norm is the maximum absolute row sum of the matrix.
 *
 * @param m The input matrix.
 * @return double The infinity norm of the matrix.
 */
double infinity_norm(matrix m) {
    double max_sum = 0.0;
    for (int i = 0; i < m.rows; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < m.cols; j++) {
            row_sum += fabs(m.data[i][j]);
        }
        if (row_sum > max_sum) {
            max_sum = row_sum;
        }
    }
    return max_sum;
}

/**
 * @brief Performs Gaussian elimination on a matrix.
 *
 * This function transforms the matrix into row echelon form using Gaussian elimination.
 * It modifies the matrix in place.
 *
 * @param m Pointer to the matrix to be transformed.
 */
void gaussian_elimination(matrix *m) {
    int lead = 0;
    for (int r = 0; r < m->rows; r++) {
        if (lead >= m->cols) return;
        int i = r;
        while (fabs(m->data[i][lead]) < TOLERANCE) {
            i++;
            if (i == m->rows) {
                i = r;
                lead++;
                if (lead == m->cols) return;
            }
        }
        double *temp = m->data[i];
        m->data[i] = m->data[r];
        m->data[r] = temp;
        for (i = r + 1; i < m->rows; i++) {
            if (fabs(m->data[r][lead]) < TOLERANCE) continue;
            double factor = m->data[i][lead] / m->data[r][lead];
            for (int j = lead; j < m->cols; j++) {
                m->data[i][j] -= factor * m->data[r][j];
            }
        }
        lead++;
    }
}

/**
 * @brief Extracts a minor matrix by removing a specified row and column.
 *
 * This function creates a new matrix by removing the specified row and column from the input matrix.
 *
 * @param m The input matrix.
 * @param row The row to be removed.
 * @param col The column to be removed.
 * @return matrix The minor matrix.
 */
matrix get_minor(matrix m, int row, int col) {
    matrix minor = create_matrix(m.rows - 1, m.cols - 1);
    if (minor.data == NULL) {
        return minor;
    }
    int minor_row = 0, minor_col = 0;
    for (int i = 0; i < m.rows; i++) {
        if (i == row) continue;
        minor_col = 0;
        for (int j = 0; j < m.cols; j++) {
            if (j == col) continue;
            minor.data[minor_row][minor_col] = m.data[i][j];
            minor_col++;
        }
        minor_row++;
    }
    return minor;
}

/**
 * @brief Splits a matrix into four quadrants.
 *
 * This function divides the input matrix into four equal-sized quadrants and stores them in the provided pointers.
 *
 * @param m The input matrix (must be square with even dimensions).
 * @param a11 Pointer to store the top-left quadrant.
 * @param a12 Pointer to store the top-right quadrant.
 * @param a21 Pointer to store the bottom-left quadrant.
 * @param a22 Pointer to store the bottom-right quadrant.
 */
void split_matrix(matrix m, matrix *a11, matrix *a12, matrix *a21, matrix *a22) {
    int n = m.rows / 2;

    *a11 = create_matrix(n, n);
    *a12 = create_matrix(n, n);
    *a21 = create_matrix(n, n);
    *a22 = create_matrix(n, n);

    if (a11->data == NULL || a12->data == NULL || a21->data == NULL || a22->data == NULL) {
        return;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            (*a11).data[i][j] = m.data[i][j];
            (*a12).data[i][j] = m.data[i][j + n];
            (*a21).data[i][j] = m.data[i + n][j];
            (*a22).data[i][j] = m.data[i + n][j + n];
        }
    }
}

/**
 * @brief Combines four matrices into one larger matrix.
 *
 * This function takes four matrices and combines them into a single larger matrix, placing them in the four quadrants.
 *
 * @param c11 The top-left quadrant.
 * @param c12 The top-right quadrant.
 * @param c21 The bottom-left quadrant.
 * @param c22 The bottom-right quadrant.
 * @return matrix The combined matrix.
 */
matrix combine_matrix(matrix c11, matrix c12, matrix c21, matrix c22) {
    int n = c11.rows;
    matrix c = create_matrix(2 * n, 2 * n);
    if (c.data == NULL) {
        return c;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c.data[i][j] = c11.data[i][j];
            c.data[i][j + n] = c12.data[i][j];
            c.data[i + n][j] = c21.data[i][j];
            c.data[i + n][j + n] = c22.data[i][j];
        }
    }
    return c;
}

/**
 * @brief Finds the next power of two greater than or equal to a given number.
 *
 * This function calculates the smallest power of two that is greater than or equal to the input number.
 *
 * @param n The input number.
 * @return int The next power of two.
 */
int next_power_of_two(int n) {
    if (n <= 0) return 1;
    int k = 1;
    while (k < n) k *= 2;
    return k;
}

/**
 * @brief Pads a matrix with zeros to a specified size.
 *
 * This function creates a new matrix of the specified size and copies the input matrix into the top-left corner,
 * padding the remaining elements with zeros.
 *
 * @param m The input matrix.
 * @param new_rows The number of rows in the padded matrix.
 * @param new_cols The number of columns in the padded matrix.
 * @return matrix The padded matrix.
 */
matrix pad_matrix(matrix m, int new_rows, int new_cols) {
    matrix padded = create_matrix(new_rows, new_cols);
    if (padded.data == NULL) {
        return padded;
    }
    for (int i = 0; i < new_rows; i++) {
        for (int j = 0; j < new_cols; j++) {
            if (i < m.rows && j < m.cols) {
                padded.data[i][j] = m.data[i][j];
            } else {
                padded.data[i][j] = 0.0;
            }
        }
    }
    return padded;
}

/**
 * @brief Implements Strassen's algorithm for matrix multiplication.
 *
 * This function multiplies two square matrices using Strassen's algorithm
 * It recursively splits the matrices into quadrants and performs seven multiplications instead of eight.
 *
 * @param a The first matrix (must be square).
 * @param b The second matrix (must be square and same size as a).
 * @param result Pointer to a matrix where the product will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if matrices are not compatible,
 *         or error codes from matrix operations.
 */
int multiply_matrices_strassen(matrix a, matrix b, matrix *result) {
    if (a.cols != b.rows) {
        printf("%sError: incompatible dimensions for multiplication.\n%s", URED, COLOR_RESET);
        return INVALID_DIMENSIONS;
    }

    int n = a.rows;
    if (n != a.cols || n != b.rows || n != b.cols) {
        return INVALID_DIMENSIONS; // Strassen assumes square matrices of equal size
    }

    if (n <= STRASSEN_THRESHOLD) {
        return multiply_matrices(a, b, result);
    }

    // Initialize all matrices
    matrix a11 = {0, 0, NULL}, a12 = {0, 0, NULL}, a21 = {0, 0, NULL}, a22 = {0, 0, NULL};
    matrix b11 = {0, 0, NULL}, b12 = {0, 0, NULL}, b21 = {0, 0, NULL}, b22 = {0, 0, NULL};
    matrix p1 = {0, 0, NULL}, p2 = {0, 0, NULL}, p3 = {0, 0, NULL}, p4 = {0, 0, NULL};
    matrix p5 = {0, 0, NULL}, p6 = {0, 0, NULL}, p7 = {0, 0, NULL};
    matrix c11 = {0, 0, NULL}, c12 = {0, 0, NULL}, c21 = {0, 0, NULL}, c22 = {0, 0, NULL};
    matrix temp1 = {0, 0, NULL}, temp2 = {0, 0, NULL}, temp3 = {0, 0, NULL};
    int status = SUCCESS;

    split_matrix(a, &a11, &a12, &a21, &a22);
    split_matrix(b, &b11, &b12, &b21, &b22);

    if (a11.data == NULL || a12.data == NULL || a21.data == NULL || a22.data == NULL ||
        b11.data == NULL || b12.data == NULL || b21.data == NULL || b22.data == NULL) {
        status = INVALID_INPUT;
        goto cleanup;
    }

    // p1 = a11 * (b12 - b22)
    status = subtract_matrices(b12, b22, &temp1);
    if (status != SUCCESS) goto cleanup;
    status = multiply_matrices_strassen(a11, temp1, &p1);
    free_matrix(&temp1);
    if (status != SUCCESS) goto cleanup;

    // p2 = (a11 + a12) * b22
    status = add_matrices(a11, a12, &temp1);
    if (status != SUCCESS) goto cleanup;
    status = multiply_matrices_strassen(temp1, b22, &p2);
    free_matrix(&temp1);
    if (status != SUCCESS) goto cleanup;

    // p3 = (a21 + a22) * b11
    status = add_matrices(a21, a22, &temp1);
    if (status != SUCCESS) goto cleanup;
    status = multiply_matrices_strassen(temp1, b11, &p3);
    free_matrix(&temp1);
    if (status != SUCCESS) goto cleanup;

    // p4 = a22 * (b21 - b11)
    status = subtract_matrices(b21, b11, &temp1);
    if (status != SUCCESS) goto cleanup;
    status = multiply_matrices_strassen(a22, temp1, &p4);
    free_matrix(&temp1);
    if (status != SUCCESS) goto cleanup;

    // p5 = (a11 + a22) * (b11 + b22)
    status = add_matrices(a11, a22, &temp1);
    if (status != SUCCESS) goto cleanup;
    status = add_matrices(b11, b22, &temp2);
    if (status != SUCCESS) {
        free_matrix(&temp1);
        goto cleanup;
    }
    status = multiply_matrices_strassen(temp1, temp2, &p5);
    free_matrix(&temp1);
    free_matrix(&temp2);
    if (status != SUCCESS) goto cleanup;

    // p6 = (a12 - a22) * (b21 + b22)
    status = subtract_matrices(a12, a22, &temp1);
    if (status != SUCCESS) goto cleanup;
    status = add_matrices(b21, b22, &temp2);
    if (status != SUCCESS) {
        free_matrix(&temp1);
        goto cleanup;
    }
    status = multiply_matrices_strassen(temp1, temp2, &p6);
    free_matrix(&temp1);
    free_matrix(&temp2);
    if (status != SUCCESS) goto cleanup;

    // p7 = (a11 - a21) * (b11 + b12)
    status = subtract_matrices(a11, a21, &temp1);
    if (status != SUCCESS) goto cleanup;
    status = add_matrices(b11, b12, &temp2);
    if (status != SUCCESS) {
        free_matrix(&temp1);
        goto cleanup;
    }
    status = multiply_matrices_strassen(temp1, temp2, &p7);
    free_matrix(&temp1);
    free_matrix(&temp2);
    if (status != SUCCESS) goto cleanup;

    // c11 = p5 + p4 - p2 + p6
    status = add_matrices(p5, p4, &temp1);
    if (status != SUCCESS) goto cleanup;
    status = subtract_matrices(temp1, p2, &temp2);
    free_matrix(&temp1);
    if (status != SUCCESS) goto cleanup;
    status = add_matrices(temp2, p6, &c11);
    free_matrix(&temp2);
    if (status != SUCCESS) goto cleanup;

    // c12 = p1 + p2
    status = add_matrices(p1, p2, &c12);
    if (status != SUCCESS) goto cleanup;

    // c21 = p3 + p4
    status = add_matrices(p3, p4, &c21);
    if (status != SUCCESS) goto cleanup;

    // c22 = p1 + p5 - p3 - p7
    status = add_matrices(p1, p5, &temp1);
    if (status != SUCCESS) goto cleanup;
    status = subtract_matrices(temp1, p3, &temp2);
    free_matrix(&temp1);
    if (status != SUCCESS) goto cleanup;
    status = subtract_matrices(temp2, p7, &c22);
    free_matrix(&temp2);
    if (status != SUCCESS) goto cleanup;

    *result = combine_matrix(c11, c12, c21, c22);
    if (result->data == NULL) {
        status = INVALID_INPUT;
    } else {
        status = SUCCESS;
    }

cleanup:
    if (a11.data != NULL) free_matrix(&a11);
    if (a12.data != NULL) free_matrix(&a12);
    if (a21.data != NULL) free_matrix(&a21);
    if (a22.data != NULL) free_matrix(&a22);
    if (b11.data != NULL) free_matrix(&b11);
    if (b12.data != NULL) free_matrix(&b12);
    if (b21.data != NULL) free_matrix(&b21);
    if (b22.data != NULL) free_matrix(&b22);
    if (p1.data != NULL) free_matrix(&p1);
    if (p2.data != NULL) free_matrix(&p2);
    if (p3.data != NULL) free_matrix(&p3);
    if (p4.data != NULL) free_matrix(&p4);
    if (p5.data != NULL) free_matrix(&p5);
    if (p6.data != NULL) free_matrix(&p6);
    if (p7.data != NULL) free_matrix(&p7);
    if (c11.data != NULL) free_matrix(&c11);
    if (c12.data != NULL) free_matrix(&c12);
    if (c21.data != NULL) free_matrix(&c21);
    if (c22.data != NULL) free_matrix(&c22);
    if (temp1.data != NULL) free_matrix(&temp1);
    if (temp2.data != NULL) free_matrix(&temp2);
    if (temp3.data != NULL) free_matrix(&temp3);
    return status;
}

/**
 * @brief A padded version of Strassen's algorithm for matrix multiplication.
 *
 * This function allows multiplication of non-square matrices by padding them to the next power of two.
 * It uses Strassen's algorithm on the padded matrices and then extracts the relevant part of the result.
 *
 * @param a The first matrix.
 * @param b The second matrix.
 * @param result Pointer to a matrix where the product will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if matrices are not compatible,
 *         or error codes from matrix operations.
 */
int multiply_matrices_strassen_padded(matrix a, matrix b, matrix *result) {
    if (a.cols != b.rows) {
        printf("%sError: incompatible dimensions for multiplication.\n%s", URED, COLOR_RESET);
        return INVALID_DIMENSIONS;
    }

    int m = a.rows;
    int n = a.cols;
    int p = b.cols;
    int max_dim = m;
    if (n > max_dim) max_dim = n;
    if (p > max_dim) max_dim = p;

    int k = next_power_of_two(max_dim);

    matrix a_padded = pad_matrix(a, k, k);
    matrix b_padded = pad_matrix(b, k, k);
    if (a_padded.data == NULL || b_padded.data == NULL) {
        free_matrix(&a_padded);
        free_matrix(&b_padded);
        return INVALID_INPUT;
    }

    matrix c_padded;
    int status = multiply_matrices_strassen(a_padded, b_padded, &c_padded);
    if (status != SUCCESS) {
        free_matrix(&a_padded);
        free_matrix(&b_padded);
        return status;
    }

    *result = create_matrix(m, p);
    if (result->data == NULL) {
        free_matrix(&a_padded);
        free_matrix(&b_padded);
        free_matrix(&c_padded);
        return INVALID_INPUT;
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            result->data[i][j] = c_padded.data[i][j];
        }
    }

    free_matrix(&a_padded);
    free_matrix(&b_padded);
    free_matrix(&c_padded);
    return SUCCESS;
}

/**
 * @brief Calculates the condition number of a matrix using a specified norm.
 *
 * The condition number is computed as the product of the norm of the matrix and the norm of its inverse.
 * It measures the sensitivity of the matrix to numerical operations.
 *
 * @param m The input matrix (must be square).
 * @param norm_func A function pointer to the norm function to be used (e.g., one_norm, infinity_norm).
 * @return double The condition number, or -1.0 if there is an error.
 */
double condition_number(matrix m, double (*norm_func)(matrix)) {
    if (m.rows != m.cols) {
        printf("%sError: condition number is only defined for square matrices.\n%s", URED, COLOR_RESET);
        return -1.0;
    }
    
    matrix inv;
    int status = inverse_matrix(m, &inv);
    if (status != SUCCESS) {
        printf("%sError: cannot compute inverse matrix (matrix may be singular).\n%s", URED, COLOR_RESET);
        return -1.0;
    }

    double norm_m = norm_func(m);
    double norm_inv = norm_func(inv);
    free_matrix(&inv);

    return norm_m * norm_inv;
}