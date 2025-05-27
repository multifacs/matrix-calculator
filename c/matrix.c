/* 
 * This file contains functions for matrix operations, including creation, input, output,
 * basic operations, decompositions, and property checks.
 */

#include <stdio.h>      // For input and output functions
#include <stdlib.h>     // For dynamic memory allocation
#include <string.h>     // For string operations
#include <math.h>       // For mathematical functions like sqrt, fabs
#include <time.h>       // For random number generation
#include "matrix.h"     // Custom header for matrix structure and function declarations

// Constants for tolerance and color codes
#define TOLERANCE       1e-10           // Tolerance for floating-point comparisons

#define URED            "\e[4;31m"      // Red underlined for errors
#define UGRN            "\e[4;32m"      // Green underlined for results
#define UYEL            "\e[4;33m"      // Yellow underlined for information/warnings
#define UBLU            "\e[4;34m"      // Blue underlined for inputs
#define UCYN            "\e[4;36m"      // Cyan underlined for menus
#define COLOR_RESET     "\e[0m"         // Reset color formatting

/* 
 * Creates a matrix with the specified number of rows and columns.
 * Matrices can be of varying sizes with dynamic allocation
 */
matrix create_matrix(int rows, int cols) {
    matrix m = {0, 0, NULL};    // Initialize matrix structure
    m.rows = rows;
    m.cols = cols;
    m.data = (double**)malloc(rows * sizeof(double *));     // Allocate memory for row pointer      
    if (m.data == NULL) {
        fprintf(stderr, "%sError: Failed to allocate memory for matrix rows.\n%s", URED, COLOR_RESET);
        return m;   // Return matrix with NULL data to indicate failure
    }
    for (int i = 0; i < rows; i++) {
        m.data[i] = (double *)malloc(cols * sizeof(double));    //Allocate memory for each row
        if (m.data[i] == NULL) {
            fprintf(stderr, "%sError: Failed to allocate memory for matrix row %d.\n%s", URED, i, COLOR_RESET);
            for (int k = 0; k < i; k++) free(m.data[k]);        // Free previously allocated rows
            free(m.data);
            m.data = NULL;
            return m;   // Return matrix with NULL data to indicate failure
        }
    }
    return m;      // Return the created matrix
}
 
// Frees the memory allocated for the matrix to prevent memory leaks when matrices are no longer needed.
void free_matrix(matrix *m) {
    for (int i = 0; i < m -> rows; i++) {
        free(m -> data[i]);     // Free each row
    }
    free(m -> data);    // Free the array of row pointers
    m -> data = NULL;
    m -> rows = 0;
    m -> cols = 0;
}


// Inputs matrix elements from the user to ensure each element is correctly entered and validated.
void input_matrix(matrix *m) {
    printf("\n%sInput matrix elements (%d x %d):\n%s", UBLU, m -> rows, m -> cols, COLOR_RESET);
    for (int i = 0; i < m -> rows; i++) {
        for (int j = 0; j < m -> cols; j++) {
            int valid_input = 0;
            while (!valid_input) {
                printf("%sElement [%d][%d]: %s", UBLU, i, j, COLOR_RESET);
                if (scanf("%lf", &m -> data[i][j]) == 1) {
                    valid_input = 1;        // Input is valid if scanf successfully reads a double
                } else {
                    printf("%sInvalid input. Please enter a number.\n%s", URED, COLOR_RESET);
                    while (getchar() != '\n');      // Clear input buffer
                }
            }
        }
    }
    while (getchar() != '\n');      // Clear any remaining input
}

// Inputs matrix elements from the user with validation. Ensures only numeric values are accepted, prompting for re-entry on invalid input.
void print_matrix(matrix m)
{
    printf("\n%sMatrix [%d x %d]:\n%s", UYEL, m.rows, m.cols, COLOR_RESET);

    int max_width = 0;
    char buffer[50];
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            double val = m.data[i][j];
            if (fabs(fmod(val, 1.0)) < TOLERANCE) {     // Check if value is effectively an integer
                snprintf(buffer, sizeof(buffer), "%lld", (long long)val);
            } else {
                snprintf(buffer, sizeof(buffer), "%.2f", val);      // Use two decimal places
            }
            int len = strlen(buffer);
            if (len > max_width) {
                max_width = len;        //Find the maximum width for aligment
            }
        }
    }

    // Print top border
    printf("+");
    for (int j = 0; j < m.cols; j++) {
        for (int k = 0; k < max_width + 2; k++) {
            printf("-");
        }
    }
    printf("+\n");

    // Print matrix rows
    for (int i = 0; i < m.rows; i++) {
        printf("|");
        for (int j = 0; j < m.cols; j++) {
            double val = m.data[i][j];
            if (fabs(fmod(val, 1.0)) < TOLERANCE) {
                printf(" %*d ", max_width, (int)val);
            } else {
                printf(" %*.2f ", max_width, val);
            }
        }
        printf("|\n");
    }

    // Print bottom border
    printf("+");
    for (int j = 0; j < m.cols; j++) {
        for (int k = 0; k < max_width + 2; k++) {
            printf("-");
        }
    }
    printf("+\n");
}

/* 
 * Allows interactive editing of matrix elements after initial input.
 * Displays the matrix and prompts the user to edit elements by specifying coordinates.
 */
void edit_matrix(matrix *m) {
    char input[10];     // Create buffer for input string
    int valid_choice = 0;

    do {
        print_matrix(*m);   // Print current matrix
        printf("Do you want to edit any element? (y/n): ");

        // Read input
        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("%sError reading input.%s\n", URED, COLOR_RESET);
            continue;
        }

        // Delete \n from input
        input[strcspn(input, "\n")] = 0;

        // Check for one char in input
        if (strlen(input) == 1) {
            char choice = input[0];
            if (choice == 'y' || choice == 'Y') {
                int row, col;
                double new_value;

                printf("Enter row (0 to %d): ", m -> rows - 1);
                scanf("%d", &row);

                printf("Enter column (0 to %d): ", m -> cols - 1);
                scanf("%d", &col);

                if (row >= 0 && row < m -> rows && col >= 0 && col < m -> cols) {
                    printf("Enter new value for element [%d][%d]: ", row, col);
                    scanf("%lf", &new_value);
                    m -> data[row][col] = new_value;        // Update element of matrix
                } else {
                    printf("%sInvalid row or column index.%s\n", URED, COLOR_RESET);
                }

                // Clear buffer after scanf
                while (getchar() != '\n');
            } else if (choice == 'n' || choice == 'N') {
                valid_choice = 1;   // Exit the loop
            } else {
                printf("%sInvalid choice. Please enter 'y' or 'n'.%s\n", URED, COLOR_RESET);
            }
        } else {
            printf("%sInvalid input. Please enter a single character ('y' ofr 'n').%s\n", URED, COLOR_RESET);
        }
    } while (!valid_choice);    // Repeat until input is valid
}

//Generates a random matrix with elements in the specified range (for testing and generating sample data)
matrix generate_random_matrix(int rows, int cols, double min_val, double max_val) {
    matrix m = create_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m.data[i][j] = min_val + (max_val - min_val) * ((double)rand() / RAND_MAX);
        }
    }
    return m;
}

// Checks if two matrices are equal within a tolerance
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

// Creates an identity matrix of the specified size
matrix create_identity_matrix(int size) {
    matrix identity = create_matrix(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            identity.data[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    return identity;
}

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

// Function for adding two matrices
matrix add_matrices(matrix a, matrix b)
{
    if (a.rows != b.rows || a.cols != b.cols)
    {
        printf("%sError: matrices must have the same dimensions for addition.\n%s", URED, COLOR_RESET);
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
        printf("%sError: matrices must have the same dimensions for subtraction.\n%s", URED, COLOR_RESET);
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
        printf("%sError: number of columns in first matrix must equal number of rows in second matrix for multiplication.\n%s", URED, COLOR_RESET);
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

// Function for matrix multiplication using Strassen's algorithm
matrix multiply_matrices_strassen(matrix a, matrix b) {

    // Base case: if matrix 1x1, perform simple multiplication
    if (a.rows == 1) {
        matrix result = create_matrix (1, 1);
        result.data[0][0] = a.data[0][0] * b.data[0][0];
        return result;
    }

    int n = a.rows;
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

    // Free temporary matrtices
    free_matrix(&a11); free_matrix(&a12); free_matrix(&a21); free_matrix(&a22);
    free_matrix(&b11); free_matrix(&b12); free_matrix(&b21); free_matrix(&b22);
    free_matrix(&p1); free_matrix(&p2); free_matrix(&p3); free_matrix(&p4); free_matrix(&p5); free_matrix(&p6); free_matrix(&p7);
    free_matrix(&c11); free_matrix(&c12); free_matrix(&c21); free_matrix(&c22);

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
    matrix temp = create_matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            temp.data[i][j] = m.data[i][j];     // Copy matrix to avoid modifying the original
        }
    }
    double det = 1.0;
    int swaps = 0;
    for (int i = 0; i < temp.rows; i++) {
        int pivot_row = i;
        while (pivot_row < temp.rows && temp.data[pivot_row][i] == 0.0) {
            pivot_row++;    // Find a non-zero pivot
        }
        if (pivot_row == temp.rows) {
            free_matrix(&temp);
            return 0;   // Matrix is singular if no pivot is found
        }
        if (pivot_row != i) {
            for (int j = 0; j < temp.cols; j++) {
                double t = temp.data[i][j];
                temp.data[i][j] = temp.data[pivot_row][j];
                temp.data[pivot_row][j] = t;    // Swap rows
            }
            swaps++;    // Track swaps to adjust determinant sign
        }
        double pivot = temp.data[i][i];
        det *= pivot;   //Multiply diagonal elements into determinant
        for (int k = i + 1; k < temp.rows; k++) {
            if (pivot == 0.0) continue;     // Skip if pivot is zero
            double factor = temp.data[k][i] / pivot;
            for (int j = i; j < temp.cols; j++) {
                temp.data[k][j] -= factor * temp.data[i][j];    // Eliminate below pivot
            }
        }
    }
    free_matrix(&temp);
    return (swaps % 2 == 0) ? det : -det;   // Adjust sign based on number of swaps
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

/* 
 * Computes the inverse of the matrix using Gauss-Jordan elimination.
 * Directly computes the inverse by transforming [A|I] to [I|A^-1], avoiding separate system solving.
 * Augments the matrix with an identity matrix, then applies row operations to reduce the left side to identity.
 */
matrix inverse_matrix(matrix m)
{
    if (m.rows != m.cols) {
        printf("%sError: inverse is only defined for square matrices.\n%s", URED, COLOR_RESET);
        exit(1);
    }
    if (determinant(m) == 0) {
        printf("%sError: matrix is singular and cannot be inverted.\n%s", URED, COLOR_RESET);
        exit(1);
    }
    matrix aug = create_matrix(m.rows, 2 * m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            aug.data[i][j] = m.data[i][j];
            aug.data[i][j + m.cols] = (i == j) ? 1 : 0;     // Augment with identity matrix
        }
    }
    for (int i = 0; i < m.rows; i++) {
        double pivot = aug.data[i][i];
        if (pivot == 0) continue;   // Double check if determinant is non-zero
        for (int j = 0; j < 2 * m.cols; j++) {
            aug.data[i][j] /= pivot;    // Normalize pivot row
        }
        for (int k = 0; k < m.rows; k++) {
            if (k != i) {
                double factor = aug.data[k][i];
                for (int j = 0; j < 2 * m.cols; j++) {
                    aug.data[k][j] -= factor * aug.data[i][j];  // Eliminate column
                }
            }
        }
    }
    matrix inv = create_matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            inv.data[i][j] = aug.data[i][j + m.cols];   // Extract inverse from right half
        }
    }
    free_matrix(&aug);
    return inv;
}

// Solves system of linear equations Ax = b
matrix solve_system(matrix A, matrix b)
{
    if (A.rows != b.rows || b.cols != 1) {
        printf("%sError: invalid dimensions for system solving.\n%s", URED, COLOR_RESET);
        exit(1);
    }
    matrix aug = create_matrix(A.rows, A.cols + 1);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            aug.data[i][j] = A.data[i][j];
        }
        aug.data[i][A.cols] = b.data[i][0];     // Augment with b vector
    }
    gaussian_elimination(&aug);
    matrix x = create_matrix(A.cols, 1);
    for (int i = A.rows - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < A.cols; j++) {
            sum += aug.data[i][j] * x.data[j][0];
        }
        if (aug.data[i][i] == 0) continue;      // Skip if no solution (singular case)
        x.data[i][0] = (aug.data[i][A.cols] - sum) / aug.data[i][i];    // Back substitution
    }
    free_matrix(&aug);
    return x;
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
        printf("%sError: matrix must be square for exponentiation.\n%s", URED, COLOR_RESET);
        exit(1);
    }

    if (exponent == 0) {
        return create_identity_matrix(m.rows);      // Any matrix to power 0 is the identity matrix
    } else if (exponent > 0) {
        matrix result = create_identity_matrix(m.rows);
        for (int i = 0; i < exponent; i++) {
            matrix temp = multiply_matrices(result, m);
            free_matrix(&result);
            result = temp;      // Repeated multiplication for positive powers
        }
        return result;
    } else {
        if (determinant(m) == 0) {
            printf("%sError: matrix is singular and cannot be raised to a negative power.\n%s", URED, COLOR_RESET);
            exit(1);
        }

        matrix m_inv = inverse_matrix(m);
        matrix result = create_identity_matrix(m.rows);
        for (int i = 0; i < -exponent; i++) {
            matrix temp = multiply_matrices(result, m_inv);
            free_matrix(&result);
            result = temp;      // Repeated multiplication with inverse for negative powers
        }
        free_matrix(&m_inv);
        return result;
    }
}

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
    if (norm < TOLERANCE) { // Если норма близка к нулю
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
        matrix Qk = create_identity_matrix(n);
        for (int i = k; i < n; i++) {
            for (int j = k; j < n; j++) {
                Qk.data[i][j] = H.data[i - k][j - k];
            }
        }
        matrix temp = multiply_matrices(Qk, *Q);
        free_matrix(Q);
        *Q = temp;
        matrix temp2 = multiply_matrices(Qk, *R);
        free_matrix(R);
        *R = temp2;
        free_matrix(&Qk);
        free_matrix(&H);
    }
}

// Implemets QR algorithm to compute eigenvalues and eigenvectors
// Iterativly applies QR decomposition to converge to diagonal form
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
 * Performs LU decomposition on the matrix.
 * Factorizes A into L*U.
 */
void lu_decomposition(matrix m, matrix *L, matrix *U) {
    if (m.rows != m.cols) {
        printf("%sError: matrix must be square for LU decomposition.\n%s", URED, COLOR_RESET);
        exit(1);
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
    for (int k = 0; k < n - 1; k++) {
        for (int i = k + 1; i < n; i++) {
            if ((*U).data[k][k] == 0) {
                printf("%sError: matrix is singular.\n%s", URED, COLOR_RESET);
                exit(1);
            }
            double factor = (*U).data[i][k] / (*U).data[k][k];
            (*L).data[i][k] = factor;       // Stores elimination factor for L
            for (int j = k; j < n; j++) {
                (*U).data[i][j] -= factor * (*U).data[k][j];    // Update U
            }
        }
    }
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