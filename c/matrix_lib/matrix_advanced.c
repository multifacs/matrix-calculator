#include <stdio.h>    // Для вывода ошибок (printf)
#include <stdlib.h>   // Для выделения памяти (malloc, free)
#include <math.h>     // Для математических функций (fabs, sqrt)
#include "matrix.h"   // Для структуры matrix и прототипов функций
#include "../constants.h"

/* 
 * Computes the determinant using Gaussian elimination.
 * The matrix is transformed into upper triangular form by row operations, 
 * and the determinant is the product of the diagonal elements, adjusted for row swaps.
 */
double determinant(matrix m) {
    if (m.rows != m.cols) {
        printf("Error: determinant is defined only for square matrices.\n%s", URED, COLOR_RESET);
        exit(1);
    }
    
    matrix L, U;
    int status = lu_decomposition(m, &L, &U);
    if (status == 1) {
        // Матрица сингулярна
        return 0.0;
    }

    double det = 1.0;
    for (int i = 0; i < m.rows; i++) {
        det *= U.data[i][i];
    }

    free_matrix(&L);
    free_matrix(&U);
    return det;
}

/* 
 * Computes the inverse of the matrix using Gauss-Jordan elimination.
 * Directly computes the inverse by transforming [A|I] to [I|A^-1], avoiding separate system solving.
 * Augments the matrix with an identity matrix, then applies row operations to reduce the left side to identity.
 */
matrix inverse_matrix(matrix m) {
    if (m.rows != m.cols) {
        printf("%sError: inverse is only defined for square matrices.\n%s", URED, COLOR_RESET);
        exit(1);
    }

    matrix L, U;
    int status = lu_decomposition(m, &L, &U);
    if (status == 1) {
        printf("%sError: matrix is singular and cannot be inverted.\n%s", URED, COLOR_RESET);
        exit(1);
    }

    int n = m.rows;
    matrix inv = create_matrix(n, n);

    // Solve for each column of the inverse matrix
    for (int col = 0; col < n; col++) {
        // Create the column vector e (unit vector with 1 at position col)
        matrix e = create_matrix(n, 1);
        for (int i = 0; i < n; i++) {
            e.data[i][0] = (i == col) ? 1.0 : 0.0;
        }

        // Solve Ly = e for y using forward substitution
        matrix y = create_matrix(n, 1);
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L.data[i][j] * y.data[j][0];
            }
            y.data[i][0] = (e.data[i][0] - sum) / L.data[i][i];
        }

        // Solve Ux = y for x using back substitution
        matrix x = create_matrix(n, 1);
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += U.data[i][j] * x.data[j][0];
            }
            x.data[i][0] = (y.data[i][0] - sum) / U.data[i][i];
        }

        // Copy the solution x to the inverse matrix column
        for (int i = 0; i < n; i++) {
            inv.data[i][col] = x.data[i][0];
        }

        free_matrix(&e);
        free_matrix(&y);
        free_matrix(&x);
    }

    free_matrix(&L);
    free_matrix(&U);
    return inv;
}

// Solves system of linear equations Ax = b
matrix solve_system(matrix A, matrix b) {
    int m = A.rows;
    int n = A.cols;

    if (m == n) {
        // Квадратная система: используем LU-разложение
        matrix L, U;
        int status = lu_decomposition(A, &L, &U);
        if (status == 1) {
            printf("%sError: matrix is singular and cannot solve the system.\n%s", URED, COLOR_RESET);
            exit(1);
        }

        matrix y = create_matrix(m, 1);
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L.data[i][j] * y.data[j][0];
            }
            y.data[i][0] = (b.data[i][0] - sum) / L.data[i][i];
        }

        matrix x = create_matrix(n, 1);
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += U.data[i][j] * x.data[j][0];
            }
            x.data[i][0] = (y.data[i][0] - sum) / U.data[i][i];
        }

        free_matrix(&L);
        free_matrix(&U);
        free_matrix(&y);
        return x;
    } else if (m > n) {
        // Переопределенная система: метод наименьших квадратов
        matrix At = transpose_matrix(A);
        matrix AtA = multiply_matrices(At, A);
        matrix Atb = multiply_matrices(At, b);

        // Решаем (A^T A) x = A^T b с помощью LU-разложения
        matrix L, U;
        int status = lu_decomposition(AtA, &L, &U);
        if (status == 1) {
            printf("%sError: matrix A^T A is singular and cannot solve the system.\n%s", URED, COLOR_RESET);
            free_matrix(&At);
            free_matrix(&AtA);
            free_matrix(&Atb);
            exit(1);
        }

        matrix y = create_matrix(n, 1);
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L.data[i][j] * y.data[j][0];
            }
            y.data[i][0] = (Atb.data[i][0] - sum) / L.data[i][i];
        }

        matrix x = create_matrix(n, 1);
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += U.data[i][j] * x.data[j][0];
            }
            x.data[i][0] = (y.data[i][0] - sum) / U.data[i][i];
        }

        free_matrix(&At);
        free_matrix(&AtA);
        free_matrix(&Atb);
        free_matrix(&L);
        free_matrix(&U);
        free_matrix(&y);
        return x;
    } else {
        // Недоопределенная система: псевдообратная матрица
        matrix At = transpose_matrix(A);
        matrix AAt = multiply_matrices(A, At);
        matrix inv_AAt = inverse_matrix(AAt); // Предполагается, что A A^T обратима

        matrix A_pseudo = multiply_matrices(At, inv_AAt);
        matrix x = multiply_matrices(A_pseudo, b);

        free_matrix(&At);
        free_matrix(&AAt);
        free_matrix(&inv_AAt);
        free_matrix(&A_pseudo);
        return x;
    }
}

/* 
 * Computes the rank of the matrix.
 * Reduces the matrix to row echelon form, where rank is the number of non-zero rows.
 */
int rank(matrix m)
{
    matrix temp = create_matrix(m.rows, m.cols);
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
            if (temp.data[i][j] != 0) {
                zero_row = 0;
                break;
            }
        }
        if (!zero_row) rank_count++;
    }
    free_matrix(&temp);
    return rank_count;
}

/* 
 * Raises the matrix to the specified power.
 * TODO Optimize using exponentiation by squaring to reduce complexity.
 */
matrix matrix_power(matrix m, int exponent) {
    if (m.rows != m.cols) {
        printf("%sError: matrix must be square.%s\n", URED, COLOR_RESET);
        exit(1);
    }

    if (exponent == 0) {
        return create_identity_matrix(m.rows);
    }

    matrix result = create_identity_matrix(m.rows);
    matrix base = create_matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            base.data[i][j] = m.data[i][j];
        }
    }

    int abs_exp = abs(exponent);
    while (abs_exp > 0) {
        if (abs_exp % 2 == 1) {
            matrix temp = multiply_matrices(result, base);
            free_matrix(&result);
            result = temp;
        }

        matrix temp = multiply_matrices(base, base);
        free_matrix(&base);
        base = temp;
        abs_exp /= 2;
    }

    if (exponent < 0) {
        if (fabs(determinant(result)) < TOLERANCE) {
            printf("%sError: matrix is singular and cannot be inverted.%s\n", URED, COLOR_RESET);
            free_matrix(&result);
            free_matrix(&base);
            exit(1);
        }
        matrix inv = inverse_matrix(result);
        free_matrix(&result);
        result = inv;
    }

    free_matrix(&base);
    return result;
}

// Implements QR algorithm to compute eigenvalues and eigenvectors
// Iteratively applies QR decomposition to converge to diagonal form
void qr_algorithm(matrix m, matrix *eigenvalues, matrix *eigenvectors, int max_iter, double tol) {
    if (m.rows != m.cols) {
        printf("%sError: matrix must be square.\n%s", URED, COLOR_RESET);
        exit(1);
    }
    int n = m.rows;
    matrix A = create_matrix(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A.data[i][j] = m.data[i][j];
        }
    }
    matrix Q_total = create_identity_matrix(n);
    for (int iter = 0; iter < max_iter; iter++) {
        matrix Q, R;
        qr_decomposition(A, &Q, &R);
        matrix A_new = multiply_matrices(R, Q);
        matrix Q_total_new = multiply_matrices(Q_total, Q);
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
    for (int i = 0; i < n; i++) {
        (*eigenvalues).data[i][0] = A.data[i][i];
    }
    *eigenvectors = Q_total;
    free_matrix(&A);
}

/* 
 * Computes the Frobenius norm of the matrix.
 * Common measure of matrix magnitude, akin to Euclidean norm for vectors.
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

// Computes the one-norm (maximum column sum) of the matrix.
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

// Computes the infinity norm (maximum row sum) of the matrix.
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

/* 
 * Performs Gaussian elimination on the matrix.
 * Method for solving linear systems and computing rank or determinants.
 */
void gaussian_elimination(matrix *m)
{
    int lead = 0;
    for (int r = 0; r < m -> rows; r++) {
        if (lead >= m -> cols) return;
        int i = r;
        while (m -> data[i][lead] == 0) {
            i++;
            if (i == m->rows) {
                i = r;
                lead++;
                if (lead == m -> cols) return;
            }
        }
        double *temp = m -> data[i];
        m -> data[i] = m -> data[r];
        m -> data[r] = temp;    // Swap rows to bring non-zero pivot into position
        for (i = r + 1; i < m -> rows; i++) {
            if (m -> data[r][lead] == 0) continue;
            double factor = m -> data[i][lead] / m -> data[r][lead];
            for (int j = lead; j < m -> cols; j++) {
                m -> data[i][j] -= factor * m -> data[r][j];    // Eliminate below pivot
            }
        }
        lead++;
    }
}

// Helper function to get minor matrix (submatrix excluding a row and column)
matrix get_minor(matrix m, int row, int col)
{
    matrix minor = create_matrix(m.rows - 1, m.cols - 1);
    int minor_row = 0, minor_col = 0;
    for (int i = 0; i < m.rows; i++)
    {
        if (i == row)
            continue;
        minor_col = 0;
        for (int j = 0; j < m.cols; j++)
        {
            if (j == col)
                continue;
            minor.data[minor_row][minor_col] = m.data[i][j];
            minor_col++;
        }
        minor_row++;
    }
    return minor;
}

// Helper function to split a matrix into four submatrices for Strassen's algorithm
void split_matrix(matrix m, matrix *a11, matrix *a12, matrix *a21, matrix *a22) {
    int n = m.rows / 2;     // Size of submatrices

    *a11 = create_matrix(n, n);
    *a12 = create_matrix(n, n);
    *a21 = create_matrix(n, n);
    *a22 = create_matrix(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            (*a11).data[i][j] = m.data[i][j];           // Top-left quadrant
            (*a12).data[i][j] = m.data[i][j + n];       // Top-right quadrant
            (*a21).data[i][j] = m.data[i + n][j];       // Bottom-left quadrant
            (*a22).data[i][j] = m.data[i + n][j + n];   // Bottom-right quadrant
        }
    }
}

// Helper function to combine four submatrices into one matrix for Strassen's algorithm
matrix combine_matrix(matrix c11, matrix c12, matrix c21, matrix c22) {
    int n = c11.rows;       // Size of submatrices;
    matrix c = create_matrix(2 * n, 2 * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c.data[i][j] = c11.data[i][j];                 // Top-left quadrant
            c.data[i][j + n] = c12.data[i][j];             // Top-right quadrant
            c.data[i + n][j] = c21.data[i][j];             // Bottom-left quadrant
            c.data[i + n][j + n] = c22.data[i][j];         // Bottom-right quadrant
        }
    }
    return c;
}

// Find the smallest power of two greater then or equal to n
int next_power_of_two(int n) {
    if (n <= 0) return 1;

    int k = 1;
    while (k < n) k *= 2;
    return k;
}

// Pad a matrix with zeroes to the specified size
matrix pad_matrix(matrix m, int new_rows, int new_cols) {
    matrix padded = create_matrix(new_rows, new_cols);
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

// Function for matrix multiplication using Strassen's algorithm
matrix multiply_matrices_strassen(matrix a, matrix b) {
    int n = a.rows;

    if (n <= STRASSEN_THRESHOLD) {
        return multiply_matrices(a, b);
    }

    matrix a11, a12, a21, a22;
    matrix b11, b12, b21, b22;

    // Split matrices into 4 submatrices
    split_matrix(a, &a11, &a12, &a21, &a22);
    split_matrix(b, &b11, &b12, &b21, &b22);

    // Compute the seven products recursively
    matrix p1 = multiply_matrices_strassen(a11, subtract_matrices(b12, b22));
    matrix p2 = multiply_matrices_strassen(add_matrices(a11, a12), b22);
    matrix p3 = multiply_matrices_strassen(add_matrices(a21, a22), b11);
    matrix p4 = multiply_matrices_strassen(a22, subtract_matrices(b21, b11));
    matrix p5 = multiply_matrices_strassen(add_matrices(a11, a22), add_matrices(b11, b22));
    matrix p6 = multiply_matrices_strassen(subtract_matrices(a12, a22), add_matrices(b21, b22));
    matrix p7 = multiply_matrices_strassen(subtract_matrices(a11, a21), add_matrices(b11, b12));

    // Compute the submatrices of the result
    matrix c11 = add_matrices(subtract_matrices(add_matrices(p5, p4), p2), p6);
    matrix c12 = add_matrices(p1, p2);
    matrix c21 = add_matrices(p3, p4);
    matrix c22 = subtract_matrices(add_matrices(p1, p5), add_matrices(p3, p7));

    // Combine submatrices into the final result
    matrix result = combine_matrix(c11, c12, c21, c22);

    // Free temporary matrices
    free_matrix(&a11); free_matrix(&a12); free_matrix(&a21); free_matrix(&a22);
    free_matrix(&b11); free_matrix(&b12); free_matrix(&b21); free_matrix(&b22);
    free_matrix(&p1); free_matrix(&p2); free_matrix(&p3); free_matrix(&p4); free_matrix(&p5); free_matrix(&p6); free_matrix(&p7);
    free_matrix(&c11); free_matrix(&c12); free_matrix(&c21); free_matrix(&c22);

    return result;

}

// Function to handle padding and apply Strassen multiplication for any size
matrix multiply_matrices_strassen_padded(matrix a, matrix b) {
    if (a.cols != b.rows) {
        printf("%sError: incompatible dimensions for multiplication.\n%s", URED, COLOR_RESET);
        exit(1);
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
    matrix c_padded = multiply_matrices_strassen(a_padded, b_padded);

    matrix c = create_matrix(m, p);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            c.data[i][j] = c_padded.data[i][j];
        }
    }

    free_matrix(&a_padded);
    free_matrix(&b_padded);
    free_matrix(&c_padded);
    return c;
}