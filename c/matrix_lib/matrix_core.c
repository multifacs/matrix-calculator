#include <stdio.h>    
#include <stdlib.h>  
#include <string.h>  
#include <math.h>     
#include <time.h>
#include <ctype.h>     
#include "matrix.h"  
#include "../constants.h"

// Define structure for mathematical constants
typedef struct {
    const char* input_symbol;       // String user types, e.g., "pi"
    double value;                   // Numerical value, e.g., M_PI
    const char* display_symbol;     // Display symbol, e.g., "π"
} constant;

// Array of supported mathematical constants
static constant constants[] = {
    {"pi", 3.141592653589793, "π"},
    {"e", 2.718281828459045, "e"},
    {"sqrt(2)", 1.4142135623730951, "√2"},
    {"sqrt(3)", 1.7320508075688772, "√3"},
    {"phi", 1.618033988749895, "φ"},
};
static int num_constants = sizeof(constants) / sizeof(constant);

// Helper function to trim leading and trailing spaces from input
void trim(char *str) {
    int len = strlen(str);
    int start = 0;
    while (start < len && isspace(str[start])) start++;
    int end = len - 1;
    while (end > start && isspace(str[end])) end--;
    memmove(str, str + start, end - start + 1);
    str[end - start + 1] = '\0';
}

/**
 * @brief Creates a matrix with the specified number of rows and columns.
 *
 * This function allocates memory for a matrix with the given dimensions.
 * It initializes the matrix structure and allocates memory for the data array.
 * If memory allocation fails, it prints an error message and exits the program.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @return matrix The created matrix structure.
 */
matrix create_matrix(int rows, int cols) {
    matrix m = {0, 0, NULL};
    m.rows = rows;
    m.cols = cols;

    m.data = (double**)malloc(rows * sizeof(double*));
    if (m.data == NULL) {
        fprintf(stderr, "%sError: Failed to allocate memory for matrix rows.\n%s", URED, COLOR_RESET);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        m.data[i] = (double*)malloc(cols * sizeof(double));
        if (m.data[i] == NULL) {
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

/**
 * @brief Frees the memory allocated for a matrix.
 *
 * This function deallocates the memory used by the matrix data and resets the matrix structure.
 *
 * @param m Pointer to the matrix to be freed.
 */
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

/**
 * @brief Inputs matrix elements from the user, supporting mathematical symbols.
 *
 * This function prompts the user to enter each element of the matrix.
 * It supports entering mathematical constants like "pi" or "e", which are converted to their numerical values.
 * If the input is not a recognized constant or a valid number, it prompts the user again.
 *
 * @param m Pointer to the matrix to be filled.
 */
void input_matrix(matrix *m) {
    char input[100];
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%sElement [%d][%d]: %s", UBLU, i, j, COLOR_RESET);
            fgets(input, sizeof(input), stdin);
            trim(input);
            int found = 0;
            for (int k = 0; k < num_constants; k++) {
                if (strcmp(input, constants[k].input_symbol) == 0) {
                    m->data[i][j] = constants[k].value;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                if (sscanf(input, "%lf", &m->data[i][j]) != 1) {
                    printf("%sInvalid input. Enter a number or a symbol (e.g., pi, sqrt(2)).\n%s", URED, COLOR_RESET);
                    j--;
                }
            }
        }
    }
}

/**
 * @brief Determines the display width of a string, considering special symbols.
 *
 * This helper function calculates the width needed to display a string.
 * For special mathematical symbols like "π", it returns a width of 1.
 * For regular strings, it returns the length of the string.
 *
 * @param str The string to calculate the width for.
 * @return int The display width of the string.
 */
int get_display_width(const char* str) {
    if (strcmp(str, "π") == 0) return 1;  
    if (strcmp(str, "e") == 0) return 1; 
    if (strcmp(str, "φ") == 0) return 1;  
    return strlen(str);                  
}

/**
 * @brief Prints the matrix in a formatted way, handling mathematical symbols.
 *
 * This function displays the matrix with proper alignment and formatting.
 * It checks if matrix elements correspond to known mathematical constants and displays their symbols if they do.
 * Otherwise, it displays the numerical value, formatting integers and floats appropriately.
 *
 * @param m The matrix to be printed.
 */
void print_matrix(matrix m) {
    printf("\n%sMatrix [%d x %d]:\n%s", UYEL, m.rows, m.cols, COLOR_RESET);

    int max_width = 0;
    char buffer[50];
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            double val = m.data[i][j];
            int is_constant = 0;
            for (int k = 0; k < num_constants; k++) {
                if (fabs(val - constants[k].value) < TOLERANCE) {
                    snprintf(buffer, sizeof(buffer), "%s", constants[k].display_symbol);
                    is_constant = 1;
                    break;
                }
            }
            if (!is_constant) {
                if (fabs(val) < DISPLAY_TOL) {
                    snprintf(buffer, sizeof(buffer), "0");
                } else if (fabs(fmod(val, 1.0)) < TOLERANCE) {
                    snprintf(buffer, sizeof(buffer), "%lld", (long long)val);
                } else {
                    snprintf(buffer, sizeof(buffer), "%.2f", val);
                }
            }
            int len = get_display_width(buffer);
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
            int is_constant = 0;
            for (int k = 0; k < num_constants; k++) {
                if (fabs(val - constants[k].value) < TOLERANCE) {
                    snprintf(buffer, sizeof(buffer), "%s", constants[k].display_symbol);
                    is_constant = 1;
                    break;
                }
            }
            if (!is_constant) {
                if (fabs(val) < DISPLAY_TOL) {
                    snprintf(buffer, sizeof(buffer), "0");
                } else if (fabs(fmod(val, 1.0)) < TOLERANCE) {
                    snprintf(buffer, sizeof(buffer), "%lld", (long long)val);
                } else {
                    snprintf(buffer, sizeof(buffer), "%.2f", val);
                }
            }
            printf(" %*s ", max_width, buffer);
        }
        printf("|\n");
    }

    printf("+");
    for (int j = 0; j < m.cols; j++) {
        for (int k = 0; k < max_width + 2; k++) printf("-");
    }
    printf("+\n");
}

/**
 * @brief Generates a matrix with random elements within a specified range.
 *
 * This function creates a matrix with the given number of rows and columns,
 * filling it with random double values between min_val and max_val.
 *
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @param min_val The minimum value for random elements.
 * @param max_val The maximum value for random elements.
 * @return matrix The generated random matrix.
 */
matrix generate_random_matrix(int rows, int cols, double min_val, double max_val) {
    matrix m = create_matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m.data[i][j] = min_val + (max_val - min_val) * ((double)rand() / RAND_MAX);
        }
    }
    return m;
}

/**
 * @brief Checks if two matrices are equal within a tolerance.
 *
 * This function compares two matrices element-wise. If all corresponding elements
 * are within the defined TOLERANCE, the matrices are considered equal.
 *
 * @param a The first matrix.
 * @param b The second matrix.
 * @return int 1 if matrices are equal within tolerance, 0 otherwise.
 */
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

/**
 * @brief Creates an identity matrix of the specified size.
 *
 * An identity matrix is a square matrix with ones on the diagonal and zeros elsewhere.
 *
 * @param size The number of rows and columns in the identity matrix.
 * @return matrix The created identity matrix.
 */
matrix create_identity_matrix(int size) {
    matrix identity = create_matrix(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            identity.data[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    return identity;
}

/**
 * @brief Adds two matrices and stores the result in a third matrix.
 *
 * This function performs element-wise addition of two matrices.
 * The matrices must have the same dimensions.
 *
 * @param a The first matrix.
 * @param b The second matrix.
 * @param result Pointer to the matrix where the result will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if matrices have different dimensions.
 */
int add_matrices(matrix a, matrix b, matrix *result) {
    if (a.rows != b.rows || a.cols != b.cols) {
        printf("%sError: matrices must have the same dimensions for addition.\n%s", URED, COLOR_RESET);
        return INVALID_DIMENSIONS;
    }
    *result = create_matrix(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            result->data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }
    return SUCCESS;
}

/**
 * @brief Subtracts one matrix from another and stores the result in a third matrix.
 *
 * This function performs element-wise subtraction of two matrices.
 * The matrices must have the same dimensions.
 *
 * @param a The matrix to subtract from.
 * @param b The matrix to subtract.
 * @param result Pointer to the matrix where the result will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if matrices have different dimensions.
 */
int subtract_matrices(matrix a, matrix b, matrix *result) {
    if (a.rows != b.rows || a.cols != b.cols) {
        printf("%sError: matrices must have the same dimensions for subtraction.\n%s", URED, COLOR_RESET);
        return INVALID_DIMENSIONS;
    }
    *result = create_matrix(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            result->data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }
    return SUCCESS;
}

/**
 * @brief Multiplies two matrices and stores the result in a third matrix.
 *
 * This function performs matrix multiplication. The number of columns in the first matrix
 * must equal the number of rows in the second matrix.
 *
 * @param a The first matrix.
 * @param b The second matrix.
 * @param result Pointer to the matrix where the product will be stored.
 * @return int Status code: SUCCESS if successful, INVALID_DIMENSIONS if matrices are not compatible for multiplication.
 */
int multiply_matrices(matrix a, matrix b, matrix *result) {
    if (a.cols != b.rows) {
        printf("%sError: number of columns in first matrix must equal number of rows in second matrix for multiplication.\n%s", URED, COLOR_RESET);
        return INVALID_DIMENSIONS;
    }
    *result = create_matrix(a.rows, b.cols);
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            result->data[i][j] = 0;
            for (int k = 0; k < a.cols; k++) {
                result->data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return SUCCESS;
}

/**
 * @brief Multiplies a matrix by a scalar value.
 *
 * This function scales each element of the matrix by the given scalar.
 *
 * @param m The input matrix.
 * @param scalar The scalar value to multiply by.
 * @return matrix The resulting matrix after scalar multiplication.
 */
matrix scalar_multiply(matrix m, double scalar) {
    matrix result = create_matrix(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            result.data[i][j] = m.data[i][j] * scalar;
        }
    }
    return result;
}

/**
 * @brief Computes the transpose of a matrix.
 *
 * The transpose of a matrix is obtained by swapping its rows and columns.
 *
 * @param m The input matrix.
 * @return matrix The transposed matrix.
 */
matrix transpose_matrix(matrix m) {
    matrix result = create_matrix(m.cols, m.rows);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            result.data[j][i] = m.data[i][j];
        }
    }
    return result;
}

/**
 * @brief Allows the user to edit specific elements of the matrix interactively.
 *
 * This function displays the current matrix and prompts the user to edit elements.
 * The user can choose to edit multiple elements in a loop until they decide to stop.
 *
 * @param m Pointer to the matrix to be edited.
 */
void edit_matrix(matrix *m) {
    if (m == NULL || m->data == NULL) {
        printf("%sError: Matrix is not initialized.\n%s", URED, COLOR_RESET);
        return;
    }

    char choice;
    do {
        printf("Current matrix:\n");
        print_matrix(*m); // Display the current matrix
        printf("Do you want to edit any element? (y/n): ");
        scanf(" %c", &choice); // Space before %c to ignore leading whitespace
        if (choice == 'y' || choice == 'Y') {
            int row, col;
            double new_value;
            printf("Enter row (0 to %d): ", m->rows - 1);
            scanf("%d", &row);
            printf("Enter column (0 to %d): ", m->cols - 1);
            scanf("%d", &col);
            if (row >= 0 && row < m->rows && col >= 0 && col < m->cols) {
                printf("Enter new value for element [%d][%d]: ", row, col);
                scanf("%lf", &new_value);
                m->data[row][col] = new_value; // Update the element
            } else {
                printf("%sInvalid row or column index.\n%s", URED, COLOR_RESET);
            }
        }
    } while (choice == 'y' || choice == 'Y');
}