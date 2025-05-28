#include <stdio.h>    // Для ввода-вывода (printf, scanf)
#include <stdlib.h>   // Для выделения памяти (malloc, free)
#include <string.h>   // Для работы со строками (strlen в print_matrix)
#include <math.h>     // Для математических функций (fabs, fmod)
#include <time.h>     // Для генерации случайных чисел (rand в generate_random_matrix)
#include "matrix.h"   // Для структуры matrix и прототипов функций
#include "../constants.h"

/* 
 * Creates a matrix with the specified number of rows and columns.
 * Matrices can be of varying sizes with dynamic allocation
 */
matrix create_matrix(int rows, int cols) {
    matrix m = {0, 0, NULL};
    m.rows = rows;
    m.cols = cols;

    // Выделяем память для строк
    m.data = (double**)malloc(rows * sizeof(double*));
    if (m.data == NULL) {
        fprintf(stderr, "%sError: Failed to allocate memory for matrix rows.\n%s", URED, COLOR_RESET);
        exit(EXIT_FAILURE);
    }

    // Выделяем память для каждого столбца
    for (int i = 0; i < rows; i++) {
        m.data[i] = (double*)malloc(cols * sizeof(double));
        if (m.data[i] == NULL) {
            // Освобождаем уже выделенную память перед выходом
            for (int k = 0; k < i; k++) {
                free(m.data[k]);
            }
            free(m.data);
            fprintf(stderr, "%sError: Failed to allocate memory for row %d.\n%s", URED, i, COLOR_RESET);
            exit(EXIT_FAILURE);
        }
    }
    return m;
}
 
// Frees the memory allocated for the matrix to prevent memory leaks when matrices are no longer needed.
void free_matrix(matrix *m) {
    if (m->data != NULL) {
        for (int i = 0; i < m->rows; i++) {
            free(m->data[i]);
        }
        free(m->data);
    }
    m->rows = 0;
    m->cols = 0;
    m->data = NULL;
}

// Inputs matrix elements from the user to ensure each element is correctly entered and validated.
void input_matrix(matrix *m) {
    char input[100];
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%sElement [%d][%d]: %s", UBLU, i, j, COLOR_RESET);
            // Читаем строку ввода
            fgets(input, sizeof(input), stdin);
            // Проверяем, удалось ли преобразовать ввод в число
            if (sscanf(input, "%lf", &m->data[i][j]) != 1) {
                printf("%sInvalid input. Please enter a number.\n%s", URED, COLOR_RESET);
                j--; // Повторяем ввод для текущего элемента
            }
        }
    }
}

// Prints matrix elements with consistent zero display using a unified threshold
void print_matrix(matrix m) {
    printf("\n%sMatrix [%d x %d]:\n%s", UYEL, m.rows, m.cols, COLOR_RESET);

    int max_width = 0;
    char buffer[50];
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            double val = m.data[i][j];
            if (fabs(val) < DISPLAY_TOL) {
                snprintf(buffer, sizeof(buffer), "0");
            } else if (fabs(fmod(val, 1.0)) < TOLERANCE) {
                snprintf(buffer, sizeof(buffer), "%lld", (long long)val);
            } else {
                snprintf(buffer, sizeof(buffer), "%.2f", val);
            }
            int len = strlen(buffer);
            if (len > max_width) max_width = len;
        }
    }

    printf("+");
    for (int j = 0; j < m.cols; j++) {
        for (int k = 0; k < max_width + 2; k++) printf("-");
    }
    printf("+\n");

    for (int i = 0; i < m.rows; i++) {
        printf("|");
        for (int j = 0; j < m.cols; j++) {
            double val = m.data[i][j];
            if (fabs(val) < DISPLAY_TOL) {
                printf(" %*s ", max_width, "0");
            } else if (fabs(fmod(val, 1.0)) < TOLERANCE) {
                printf(" %*d ", max_width, (int)val);
            } else {
                printf(" %*.2f ", max_width, val);
            }
        }
        printf("|\n");
    }

    printf("+");
    for (int j = 0; j < m.cols; j++) {
        for (int k = 0; k < max_width + 2; k++) printf("-");
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
            printf("%sInvalid input. Please enter a single character ('y' or 'n').%s\n", URED, COLOR_RESET);
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


