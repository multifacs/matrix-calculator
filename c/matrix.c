#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

// Function for creating matrix
matrix create_matrix(int rows, int cols)
{
    if (rows > 10 || cols > 10)
    {
        printf("Error: maximum size of matrix is 10x10.\n");
        exit(1);
    }
    matrix m;
    m.rows = rows;
    m.cols = cols;
    return m;
}

// Function for input of matrix elements
void input_matrix(matrix *m)
{
    printf("Input matrix elements (%d x %d):\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            printf("Element [%d][%d]: ", i, j);
            scanf("%d", &m->data[i][j]);
        }
    }
}

// Function for matrix output
void print_matrix(matrix m)
{
    printf("matrix [%d x %d]:\n", m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            printf("%d\t", m.data[i][j]);
        }
        printf("\n");
    }
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
matrix scalar_muliply(matrix m, int scalar)
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

// Function to compute determinant (recursive method)
int determinant(matrix m)
{
    if (m.rows != m.cols)
    {
        printf("Error: determinant is only defined for square matrices.\n");
        exit(1);
    }
    if (m.rows == 1)
        return m.data[0][0];
    if (m.rows == 2)
        return m.data[0][0] * m.data[1][1] - m.data[0][1] * m.data[1][0];

    int det = 0;
    for (int j = 0; j < m.cols; j++)
    {
        matrix minor = get_minor(m, 0, j);
        det += (j % 2 == 0 ? 1 : -1) * m.data[0][j] * determinant(minor);
    }
    return det;
}

// Helper function for Gaussian elimination
void gaussian_elimination(matrix *m)
{
    int lead = 0;
    for (int r = 0; r < m->rows; r++)
    {
        if (lead >= m->cols)
            return;
        int i = r;
        while (m->data[i][lead] == 0)
        {
            i++;
            if (i == m->rows)
            {
                i = r;
                lead++;
                if (lead == m->cols)
                    return;
            }
        }
        // Swap rows i and r
        for (int j = 0; j < m->cols; j++)
        {
            int temp = m->data[i][j];
            m->data[i][j] = m->data[r][j];
            m->data[r][j] = temp;
        }
        // Eliminate
        for (i = r + 1; i < m->rows; i++)
        {
            if (m->data[r][lead] == 0)
                continue;
            int factor = m->data[i][lead] / m->data[r][lead];
            for (int j = lead; j < m->cols; j++)
            {
                m->data[i][j] -= factor * m->data[r][j];
            }
        }
        lead++;
    }
}

// Function to find inverse matrix using Gauss-Jordan method
matrix inverse_matrix(matrix m)
{
    if (m.rows != m.cols)
    {
        printf("Error: inverse is only defined for square matrices.\n");
        exit(1);
    }
    if (determinant(m) == 0)
    {
        printf("Error: matrix is singular and cannot be inverted.\n");
        exit(1);
    }
    // Create augmented matrix [m | I]
    matrix aug = create_matrix(m.rows, 2 * m.cols);
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            aug.data[i][j] = m.data[i][j];
            aug.data[i][j + m.cols] = (i == j) ? 1 : 0;
        }
    }
    // Apply Gauss-Jordan elimination
    for (int i = 0; i < m.rows; i++)
    {
        int pivot = aug.data[i][i];
        if (pivot == 0)
            continue;
        for (int j = 0; j < 2 * m.cols; j++)
        {
            aug.data[i][j] /= pivot;
        }
        for (int k = 0; k < m.rows; k++)
        {
            if (k != i)
            {
                int factor = aug.data[k][i];
                for (int j = 0; j < 2 * m.cols; j++)
                {
                    aug.data[k][j] -= factor * aug.data[i][j];
                }
            }
        }
    }
    // Extract inverse from right half
    matrix inv = create_matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            inv.data[i][j] = aug.data[i][j + m.cols];
        }
    }
    return inv;
}

// Function to solve system of linear equations Ax = b
matrix solve_system(matrix A, matrix b)
{
    if (A.rows != b.rows || b.cols != 1)
    {
        printf("Error: invalid dimensions for system solving.\n");
        exit(1);
    }
    // Create augmented matrix [A | b]
    matrix aug = create_matrix(A.rows, A.cols + 1);
    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            aug.data[i][j] = A.data[i][j];
        }
        aug.data[i][A.cols] = b.data[i][0];
    }
    gaussian_elimination(&aug);
    // Back substitution to find solution
    matrix x = create_matrix(A.cols, 1);
    for (int i = A.rows - 1; i >= 0; i--)
    {
        int sum = 0;
        for (int j = i + 1; j < A.cols; j++)
        {
            sum += aug.data[i][j] * x.data[j][0];
        }
        if (aug.data[i][i] == 0)
            continue;
        x.data[i][0] = (aug.data[i][A.cols] - sum) / aug.data[i][i];
    }
    return x;
}

// Function to find rank of a matrix
int rank(matrix m)
{
    matrix temp = m;
    gaussian_elimination(&temp);
    int rank = 0;
    for (int i = 0; i < temp.rows; i++)
    {
        int zero_row = 1;
        for (int j = 0; j < temp.cols; j++)
        {
            if (temp.data[i][j] != 0)
            {
                zero_row = 0;
                break;
            }
        }
        if (!zero_row)
            rank++;
    }
    return rank;
}