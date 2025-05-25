#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "matrix.h"

#define TOLERANCE       1e-10

#define UBLK            "\e[4;30m"
#define URED            "\e[4;31m"
#define UGRN            "\e[4;32m"
#define UYEL            "\e[4;33m"
#define UBLU            "\e[4;34m"
#define UMAG            "\e[4;35m"
#define UCYN            "\e[4;36m"
#define COLOR_RESET     "\e[0m"

// Function for creating matrix
matrix create_matrix(int rows, int cols) {
    matrix m = {0, 0, NULL};
    m.rows = rows;
    m.cols = cols;
    m.data = (double**)malloc(rows * sizeof(double *));
    if (m.data == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for matrix rows.\n");
        return m;
    }
    for (int i = 0; i < rows; i++) {
        m.data[i] = (double *)malloc(cols * sizeof(double));
        if (m.data[i] == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for matrix row %d.\n", i);
            for (int k = 0; k < i; k++) free(m.data[k]);
            free(m.data);
            m.data = NULL;
            return m;
        }
    }
    return m;
}

// Free matrix memory
void free_matrix(matrix *m) {
    for (int i = 0; i < m -> rows; i++) {
        free(m -> data[i]);
    }
    free(m -> data);
    m -> data = NULL;
    m -> rows = 0;
    m -> cols = 0;
}


// Function for input of matrix elements
void input_matrix(matrix *m) {
    printf("Input matrix elements (%d x %d):\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("Element [%d][%d]: ", i, j);
            while (scanf("%lf", &m->data[i][j]) != 1) {
                int c;
                while ((c = getchar()) != '\n' && c != EOF);
                printf("Invalid input. Please enter a number.\n");
                printf("Element [%d][%d]: ", i, j);
            }
        }
    }
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

// Function for matrix output
void print_matrix(matrix m)
{
    printf("matrix [%d x %d]:\n", m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            double val = m.data[i][j];
            if (fabs(fmod(val, 1.0)) < TOLERANCE) {
                printf("%d\t", (int)val);
            } else {
                printf("%.2f\t", val);
            }
        }
        printf("\n");
    }
}

//Function for generation of random matrix
matrix generate_random_matrix(int rows, int cols, double min_val, double max_val) {
    matrix m = create_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m.data[i][j] = min_val + (max_val - min_val) * ((double)rand() / RAND_MAX);
        }
    }
    return m;
}

// Helper function for check if matrices are equal
int matrices_equal(matrix a, matrix b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        return 0;
    }
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            if (fabs(a.data[i][j] - b.data[i][j]) > TOLERANCE) {
                return 0;
            }
        }
    }
    return 1;
}

// Helper function for generating identity matrix
matrix create_identity_matrix(int size) {
    matrix identity = create_matrix(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            identity.data[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    return identity;
}

// Check for diagonality
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

// Check for symmetry
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

// Check for orthogonality
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

// Upper-triangularity check
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

// Lower-triangularity check
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

// Check for identity
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

// Function for adding two matrices
matrix add_matrices(matrix a, matrix b)
{
    if (a.rows != b.rows || a.cols != b.cols)
    {
        printf("Error: matrices must have the same dimensions for addition.\n");
        exit(1);
    }
    matrix result = create_matrix(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            result.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }
    return result;
}

// Function for subtracting two matrices
matrix subtract_matrices(matrix a, matrix b)
{
    if (a.rows != b.rows || a.cols != b.cols)
    {
        printf("Error: matrices must have the same dimensions for subtraction.\n");
        exit(1);
    }
    matrix result = create_matrix(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
        {
            result.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }
    return result;
}

// Function for multiplying two matrices
matrix multiply_matrices(matrix a, matrix b)
{
    if (a.cols != b.rows)
    {
        printf("Error: number of columns in first matrix must equal number of rows in second matrix for multiplication.\n");
        exit(1);
    }
    matrix result = create_matrix(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < b.cols; j++)
        {
            result.data[i][j] = 0;
            for (int k = 0; k < a.cols; k++)
            {
                result.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return result;
}

// Function for multiplying a matrix by a scalar
matrix scalar_multiply(matrix m, double scalar)
{
    matrix result = create_matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            result.data[i][j] = m.data[i][j] * scalar;
        }
    }
    return result;
}

// Function for transposing matrix
matrix transpose_matrix(matrix m)
{
    matrix result = create_matrix(m.cols, m.rows);
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            result.data[j][i] = m.data[i][j];
        }
    }
    return result;
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

// Function to compute determinant (Gauss method)
double determinant(matrix m) {
    if (m.rows != m.cols) {
        printf("Ошибка: определитель определён только для квадратных матриц.\n");
        exit(1);
    }
    matrix temp = create_matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            temp.data[i][j] = m.data[i][j];
        }
    }
    double det = 1.0;
    int swaps = 0;
    for (int i = 0; i < temp.rows; i++) {
        int pivot_row = i;
        while (pivot_row < temp.rows && temp.data[pivot_row][i] == 0.0) {
            pivot_row++;
        }
        if (pivot_row == temp.rows) {
            free_matrix(&temp);
            return 0;
        }
        if (pivot_row != i) {
            for (int j = 0; j < temp.cols; j++) {
                double t = temp.data[i][j];
                temp.data[i][j] = temp.data[pivot_row][j];
                temp.data[pivot_row][j] = t;
            }
            swaps++;
        }
        double pivot = temp.data[i][i];
        det *= pivot;
        for (int k = i + 1; k < temp.rows; k++) {
            if (pivot == 0.0) continue;
            double factor = temp.data[k][i] / pivot;
            for (int j = i; j < temp.cols; j++) {
                temp.data[k][j] -= factor * temp.data[i][j];
            }
        }
    }
    free_matrix(&temp);
    return (swaps % 2 == 0) ? det : -det;
}

// Helper function for Gaussian elimination
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
        m -> data[r] = temp;
        for (i = r + 1; i < m -> rows; i++) {
            if (m -> data[r][lead] == 0) continue;
            double factor = m -> data[i][lead] / m -> data[r][lead];
            for (int j = lead; j < m -> cols; j++) {
                m -> data[i][j] -= factor * m -> data[r][j];
            }
        }
        lead++;
    }
}

// Function to find inverse matrix using Gauss-Jordan method
matrix inverse_matrix(matrix m)
{
    if (m.rows != m.cols) {
        printf("Error: inverse is only defined for square matrices.\n");
        exit(1);
    }
    if (determinant(m) == 0) {
        printf("Error: matrix is singular and cannot be inverted.\n");
        exit(1);
    }
    matrix aug = create_matrix(m.rows, 2 * m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            aug.data[i][j] = m.data[i][j];
            aug.data[i][j + m.cols] = (i == j) ? 1 : 0;
        }
    }
    for (int i = 0; i < m.rows; i++) {
        double pivot = aug.data[i][i];
        if (pivot == 0) continue;
        for (int j = 0; j < 2 * m.cols; j++) {
            aug.data[i][j] /= pivot;
        }
        for (int k = 0; k < m.rows; k++) {
            if (k != i) {
                double factor = aug.data[k][i];
                for (int j = 0; j < 2 * m.cols; j++) {
                    aug.data[k][j] -= factor * aug.data[i][j];
                }
            }
        }
    }
    matrix inv = create_matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            inv.data[i][j] = aug.data[i][j + m.cols];
        }
    }
    free_matrix(&aug);
    return inv;
}

// Function to solve system of linear equations Ax = b
matrix solve_system(matrix A, matrix b)
{
    if (A.rows != b.rows || b.cols != 1) {
        printf("Error: invalid dimensions for system solving.\n");
        exit(1);
    }
    matrix aug = create_matrix(A.rows, A.cols + 1);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            aug.data[i][j] = A.data[i][j];
        }
        aug.data[i][A.cols] = b.data[i][0];
    }
    gaussian_elimination(&aug);
    matrix x = create_matrix(A.cols, 1);
    for (int i = A.rows - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < A.cols; j++) {
            sum += aug.data[i][j] * x.data[j][0];
        }
        if (aug.data[i][i] == 0) continue;
        x.data[i][0] = (aug.data[i][A.cols] - sum) / aug.data[i][i];
    }
    free_matrix(&aug);
    return x;
}

// Function to find rank of a matrix
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

// Matrix exponetiation
// FIXME optimize matrix_power (squaring?)
matrix matrix_power(matrix m, int exponent) {
    if (m.rows != m.cols) {
        printf("Error: matrix must be square for exponentiation.\n");
        exit(1);
    }

    if (exponent == 0) {
        return create_identity_matrix(m.rows);
    } else if (exponent > 0) {
        matrix result = create_identity_matrix(m.rows);
        for (int i = 0; i < exponent; i++) {
            matrix temp = multiply_matrices(result, m);
            free_matrix(&result);
            result = temp;
        }
        return result;
    } else {
        if (determinant(m) == 0) {
            printf("Error: matrix is singular and cannot be raised to a negative power.\n");
            exit(1);
        }

        matrix m_inv = inverse_matrix(m);
        matrix result = create_identity_matrix(m.rows);
        for (int i = 0; i < -exponent; i++) {
            matrix temp = multiply_matrices(result, m_inv);
            free_matrix(&result);
            result = temp;
        }
        free_matrix(&m_inv);
        return result;
    }
}

// Cholesky decomposition
matrix cholesky_decomposition(matrix m) {
    if (m.rows != m.cols) {
        printf("Error: matrix must be square for Cholesky decomposition.\n");
        exit(1);
    }
    if (!is_symmetric(m)) {
        printf("Error: matrix must be symmetric for Cholesky decomposition");
        exit(1);
    }

    int n = m.rows;

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
            printf("Error: matrix is not positive definite.\n");
            exit(1);
        }
    }

    matrix L = create_matrix(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L.data[i][j] = 0.0;
        }
    }
    for (int k = 0; k < n; k++) {
        double sum1 = 0.0;
        for (int m = 0; m < k; m++) {
            sum1 += L.data[k][m] * L.data[k][m];
        }
        double temp = m.data[k][k] - sum1;
        if (temp <= 0) {
            printf("Error: matrix is not positive definite.\n");
            free_matrix(&L);
            exit(1);
        }
        L.data[k][k] = sqrt(temp);
        for (int i = k + 1; i < n; i++) {
            double sum2 = 0.0;
            for (int m = 0; m < k; m++) {
                sum2 += L.data[i][m] * L.data[k][m];
            }
            L.data[i][k] = (m.data[i][k] - sum2) / L.data[k][k];
        }
    }
    return L;
}

// Power method for Eigenvalues and Eigenvectors
// FIXME QR-algorithm?
void power_method(matrix m, double *eigenvalue, matrix *eigenvector, int max_iter, double tol) {
    if (m.rows != m.cols) {
        printf("Error: matrix must be square for power method");
        exit(1);
    }

    int n = m.rows;
    matrix v = create_matrix(n, 1);
    for (int i = 0; i < n; i++) {
        v.data[i][0] = (double)rand() /RAND_MAX;
    }

    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        norm += v.data[i][0] * v.data[i][0];
    }
    norm = sqrt(norm);
    for (int i = 0; i < n; i++) {
        v.data[i][0] /= norm;
    }

    double prev_lambda = 0.0;
    for (int iter = 0; iter < max_iter; iter++) {
        matrix temp = multiply_matrices(m, v);
        double lambda = 0.0;
        for (int i = 0; i < n; i++) {
            lambda += v.data[i][0] * temp.data[i][0];
        }
        norm = 0.0;
        for (int i = 0; i < n; i++) {
            norm += temp.data[i][0] * temp.data[i][0];
        }
        norm = sqrt(norm);
        for (int i = 0; i < n; i++) {
            temp.data[i][0] /= norm;
        }
        free_matrix(&v);
        v = temp;
        if (iter > 0 && fabs(lambda - prev_lambda) < tol) {
            *eigenvalue = lambda;
            *eigenvector = v;
            return;
        }
        prev_lambda = lambda;
    }
    printf("Warning: power method did not converge.\n");
    *eigenvalue = prev_lambda;
    *eigenvector = v;
}

// LU decomposition
void lu_decomposition(matrix m, matrix *L, matrix *U) {
    if (m.rows != m.cols) {
        printf("Error: matrix must be square for LU decomposition.\n");
        exit(1);
    }

    int n = m.rows;
    *L = create_matrix(n, n);
    *U = create_matrix(n, n);
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
    for (int k = 0; k < n - 1; k++) {
        for (int i = k + 1; i < n; i++) {
            if ((*U).data[k][k] == 0) {
                printf("Error: matrix is singular.\n");
                exit(1);
            }
            double factor = (*U).data[i][k] / (*U).data[k][k];
            (*L).data[i][k] = factor;
            for (int j = k; j < n; j++) {
                (*U).data[i][j] -= factor * (*U).data[k][j];
            }
        }
    }
}

// Frobenius norm
double frobenius_norm(matrix m) {
    double sum = 0.0;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            sum += m.data[i][j] * m.data[i][j];
        }
    }
    return sqrt(sum);
}

// 1-Norm
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

// Infinity norm
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