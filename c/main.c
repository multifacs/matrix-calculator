/* 
 * Matrix Calculator is designed to perform matrix operations.
 * It supports basic actions and complex computations,
 * such as determinant, inverse matrix, decompositions, and eigenvalues. 
 */

#include <stdio.h>      // Library for input and output
#include <ctype.h>      // Functions for character operations
#include <string.h>     // String operations
#include <time.h>       // For random number generation
#include "matrix.h"     // Custom header file for matrix functions

// Definition of color codes for terminal text output
#define URED            "\e[4;31m"      // Errors
#define UGRN            "\e[4;32m"      // Results
#define UYEL            "\e[4;33m"      // Information/warnings
#define UBLU            "\e[4;34m"      // Inputs
#define UCYN            "\e[4;36m"      // Menus
#define COLOR_RESET     "\e[0m"         // Color reset

/* Display the main menu with available operations */
void show_menu() {
    printf("\n%sMATRIX CALCULATOR\n\n%s", UCYN, COLOR_RESET);
    printf("%s1. Add two matrices\n%s", UYEL, COLOR_RESET);
    printf("%s2. Subtract two matrices\n%s", UYEL, COLOR_RESET);
    printf("%s3. Multiply two matrices\n%s", UYEL, COLOR_RESET);
    printf("%s4. Multiply matrix by a scalar\n%s", UYEL, COLOR_RESET);
    printf("%s5. Transpose matrix\n%s", UYEL, COLOR_RESET);
    printf("%s6. Find determinant\n%s", UYEL, COLOR_RESET);
    printf("%s7. Find inverse matrix\n%s", UYEL, COLOR_RESET);
    printf("%s8. Solve system of linear equations\n%s", UYEL, COLOR_RESET);
    printf("%s9. Find rank of a matrix\n%s", UYEL, COLOR_RESET);
    printf("%s10. Generate random matrix\n%s", UYEL, COLOR_RESET);
    printf("%s11. Check matrix properties\n%s", UYEL, COLOR_RESET);
    printf("%s12. Matrix exponentiation\n%s", UYEL, COLOR_RESET);
    printf("%s13. Cholesky decomposition\n%s", UYEL, COLOR_RESET);
    printf("%s14. Eigenvalues and Eigenvector\n%s", UYEL, COLOR_RESET);
    printf("%s15. LU decomposition\n%s", UYEL, COLOR_RESET);
    printf("%s16. Matrix norms\n%s", UYEL, COLOR_RESET);
    printf("%s17. Save matrix to file\n%s", UYEL, COLOR_RESET);
    printf("%s18. Load matrix from file\n%s", UYEL, COLOR_RESET);
    printf("%s19. Use loaded matrix for operations\n%s", UYEL, COLOR_RESET);
    printf("%s20. Save random matrix to file\n%s", UYEL, COLOR_RESET);
    printf("%s21. Singular value decomposition (SVD)\n%s", UYEL, COLOR_RESET);
    printf("%s22. Schur decomposition\n%s", UYEL, COLOR_RESET);
    printf("%s23. Exit\n\n%s", UYEL, COLOR_RESET);
}

/* Get the user's choice from the menu with input validation */
int get_user_choice() {
    char input[100];
    while (1) {
        printf("%sEnter your choice: %s", UBLU, COLOR_RESET);
        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("%sEnd of input. Exiting.\n%s", URED, COLOR_RESET);
            exit(0);
        }
        input[strcspn(input, "\n")] = 0;
        char *endptr;
        long choice = strtol(input, &endptr, 10);
        if (*endptr == '\0' && choice >= 1 && choice <= 23) {
            return (int)choice;
        } else {
            printf("%sInvalid input. Please enter a number between 1 and 23.\n%s", URED, COLOR_RESET);
        }
    }
}

/* Request a positive integer from the user */
int input_positive_integer(const char* prompt) {
    char input[100];
    int value;
    while (1) {
        printf("%s", prompt);
        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("%sError reading input.\n%s", URED, COLOR_RESET);
            continue;
        }
        input[strcspn(input, "\n")] = 0;
        int is_valid = 1;
        for (size_t i = 0; i < strlen(input); i++) {
            if (!isdigit(input[i])) {
                is_valid = 0;
                break;
            }
        }
        if (is_valid && sscanf(input, "%d", &value) == 1 && value > 0) {
            return value;
        } else {
            printf("%sInvalid input. Please enter a positive integer (no decimals or letters).\n%s", URED, COLOR_RESET);
        }
    }
}

/* Collect parameters for generating a random matrix */
void input_random_matrix_params(int *rows, int *cols, double *min_val, double *max_val) {
    *rows = input_positive_integer("\nEnter number of rows: ");
    *cols = input_positive_integer("Enter number of cols: ");
    printf("%sEnter minimum value for elements: %s", UBLU, COLOR_RESET);
    while (scanf("%lf", min_val) != 1) {
        while (getchar() != '\n');
        printf("%sInvalid input. Please enter a number.\n%s", URED, COLOR_RESET);
        printf("%sEnter minimum value for elements: %s", UBLU, COLOR_RESET);
    }
    while (getchar() != '\n');
    printf("%sEnter maximum value for elements: %s", UBLU, COLOR_RESET);
    while (scanf("%lf", max_val) != 1 || *max_val < *min_val) {
        while (getchar() != '\n');
        if (*max_val < *min_val) {
            printf("%sMaximum value must be greater than or equal to minimum value.\n%s", URED, COLOR_RESET);
        } else {
            printf("%sInvalid input. Please enter a number.\n%s", URED, COLOR_RESET);
        }
        printf("%sEnter maximum value for elements: %s", UBLU, COLOR_RESET);
    }
    while (getchar() != '\n');
}

/* Create a new matrix by requesting dimensions and elements */
matrix input_matrix_new() {
    int rows = input_positive_integer("\nEnter number of rows: ");
    int cols = input_positive_integer("Enter number of columns: ");
    matrix m = create_matrix(rows, cols);
    input_matrix(&m);
    edit_matrix(&m);
    return m;
}

/* Display the matrix on the screen */
void display_matrix(matrix m) {
    print_matrix(m);
}

/* Request a scalar value from the user */
double input_scalar() {
    double scalar;
    printf("%sInput scalar: %s", UBLU, COLOR_RESET);
    scanf("%lf", &scalar);
    return scalar;
}

/* Wait for Enter to be pressed before continuing */
void wait_for_enter() {
    printf("%sPress Enter to continue...\n%s", UBLU, COLOR_RESET);
    while (getchar() != '\n');
}

/* Display eigenvalues and eigenvectors */
void display_eigen(matrix eigenvalues, matrix eigenvectors) {
    printf("\n%sEigenvalues:\n%s", UGRN, COLOR_RESET);
    for (int i = 0; i < eigenvalues.rows; i++) {
        printf("%f\n", eigenvalues.data[i][0]);
    }
    printf("\n%sEigenvectors:\n%s", UGRN, COLOR_RESET);
    for (int i = 0; i < eigenvectors.cols; i++) {
        printf("Eigenvector %d:\n", i + 1);
        for (int j = 0; j < eigenvectors.rows; j++) {
            printf("%f\n", eigenvectors.data[j][i]);
        }
        printf("\n");
    }
}

/* Save matrix to a text file */
void save_matrix_to_file(matrix m, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("%sError: unable to open file for save.\n%s", URED, COLOR_RESET);
        return;
    }
    fprintf(file, "%d %d\n", m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            fprintf(file, "%lf ", m.data[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
    printf("%sMatrix saved in %s.\n%s", UGRN, filename, COLOR_RESET);
}

/* Load matrix from a text file */
matrix load_matrix_from_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("%sError: cannot open file for reading.\n%s", URED, COLOR_RESET);
        return create_matrix(0, 0);
    }
    int rows, cols;
    if (fscanf(file, "%d %d", &rows, &cols) != 2) {
        printf("%sError: incorrect file format.\n%s", URED, COLOR_RESET);
        fclose(file);
        return create_matrix(0, 0);
    }
    if (rows <= 0 || cols <= 0) {
        printf("%sError: matrix size must be positive.\n%s", URED, COLOR_RESET);
        fclose(file);
        return create_matrix(0, 0);
    }
    matrix m = create_matrix(rows, cols);
    if (m.data == NULL) {
        fclose(file);
        return create_matrix(0, 0);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf", &m.data[i][j]) != 1) {
                printf("%sError: incorrect data format.\n%s", URED, COLOR_RESET);
                free_matrix(&m);
                fclose(file);
                return create_matrix(0, 0);
            }
        }
    }
    fclose(file);
    printf("%sMatrix loaded from %s.\n%s", UGRN, filename, COLOR_RESET);
    return m;
}

/* Operation functions with error handling and context passing */
int add_matrices_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m1 = input_matrix_new();
    matrix m2 = input_matrix_new();
    if (m1.rows != m2.rows || m1.cols != m2.cols) {
        printf("%sError: Matrices must have the same dimensions for addition.\n%s", URED, COLOR_RESET);
        free_matrix(&m1);
        free_matrix(&m2);
        return 1;
    }
    matrix result = add_matrices(m1, m2);
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m1);
    free_matrix(&m2);
    wait_for_enter();
    return 0;
}

int subtract_matrices_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m1 = input_matrix_new();
    matrix m2 = input_matrix_new();
    if (m1.rows != m2.rows || m1.cols != m2.cols) {
        printf("%sError: Matrices must have the same dimensions for subtraction.\n%s", URED, COLOR_RESET);
        free_matrix(&m1);
        free_matrix(&m2);
        return 1;
    }
    matrix result = subtract_matrices(m1, m2);
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m1);
    free_matrix(&m2);
    wait_for_enter();
    return 0;
}

int multiply_matrices_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m1 = input_matrix_new();
    matrix m2 = input_matrix_new();
    if (m1.cols != m2.rows) {
        printf("%sError: Number of columns in first matrix must equal number of rows in second matrix.\n%s", URED, COLOR_RESET);
        free_matrix(&m1);
        free_matrix(&m2);
        return 1;
    }
    matrix result = multiply_matrices_strassen_padded(m1, m2);
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m1);
    free_matrix(&m2);
    wait_for_enter();
    return 0;
}

int scalar_multiply_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    double scalar = input_scalar();
    matrix result = scalar_multiply(m, scalar);
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int transpose_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    matrix result = transpose_matrix(m);
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int determinant_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    if (m.rows != m.cols) {
        printf("%sError: Determinant is only defined for square matrices.\n%s", URED, COLOR_RESET);
        free_matrix(&m);
        return 1;
    }
    double det = determinant(m);
    printf("\n%sDeterminant: %lf\n%s", UGRN, det, COLOR_RESET);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int inverse_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    if (m.rows != m.cols) {
        printf("%sError: Inverse is only defined for square matrices.\n%s", URED, COLOR_RESET);
        free_matrix(&m);
        return 1;
    }
    matrix result = inverse_matrix(m);
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int solve_system_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix A = input_matrix_new();
    printf("%sInput vector B:\n%s", UBLU, COLOR_RESET);
    matrix B = input_matrix_new();
    if (A.rows != B.rows || B.cols != 1) {
        printf("%sError: B must be a column vector with %d rows.\n%s", URED, A.rows, COLOR_RESET);
        free_matrix(&A);
        free_matrix(&B);
        return 1;
    }
    matrix x = solve_system(A, B);
    display_matrix(x);
    free_matrix(&x);
    free_matrix(&A);
    free_matrix(&B);
    wait_for_enter();
    return 0;
}

int rank_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    int r = rank(m);
    printf("\n%sRank: %d\n%s", UGRN, r, COLOR_RESET);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int generate_random_matrix_operation(matrix *saved_matrix, int *matrix_loaded) {
    int rows, cols;
    double min_val, max_val;
    input_random_matrix_params(&rows, &cols, &min_val, &max_val);
    if (*matrix_loaded) free_matrix(saved_matrix);
    *saved_matrix = generate_random_matrix(rows, cols, min_val, max_val);
    *matrix_loaded = 1;
    printf("%sGenerated random matrix:\n%s", UGRN, COLOR_RESET);
    display_matrix(*saved_matrix);
    wait_for_enter();
    return 0;
}

int check_properties_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    printf("\n%sMatrix properties:\n%s", UGRN, COLOR_RESET);
    printf("%sDiagonal: %s\n%s", UGRN, is_diagonal(m) ? "Yes" : "No", COLOR_RESET);
    printf("%sSymmetric: %s\n%s", UGRN, is_symmetric(m) ? "Yes" : "No", COLOR_RESET);
    printf("%sUpper triangular: %s\n%s", UGRN, is_upper_triangular(m) ? "Yes" : "No", COLOR_RESET);
    printf("%sLower triangular: %s\n%s", UGRN, is_lower_triangular(m) ? "Yes" : "No", COLOR_RESET);
    printf("%sIdentity: %s\n%s", UGRN, is_identity(m) ? "Yes" : "No", COLOR_RESET);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int matrix_power_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    if (m.rows != m.cols) {
        printf("%sError: Matrix must be square for exponentiation.\n%s", URED, COLOR_RESET);
        free_matrix(&m);
        return 1;
    }
    int exponent;
    printf("%sEnter exponent: %s", UBLU, COLOR_RESET);
    scanf("%d", &exponent);
    matrix result = matrix_power(m, exponent);
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int cholesky_decomposition_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    if (m.rows != m.cols) {
        printf("%sError: Matrix must be square for Cholesky decomposition.\n%s", URED, COLOR_RESET);
        free_matrix(&m);
        return 1;
    }
    if (!is_symmetric(m)) {
        printf("%sError: Matrix must be symmetric.\n%s", URED, COLOR_RESET);
        free_matrix(&m);
        return 1;
    }
    matrix L = cholesky_decomposition(m);
    printf("%sLower triangular matrix L:\n%s", UGRN, COLOR_RESET);
    display_matrix(L);
    free_matrix(&L);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int eigenvalues_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    if (m.rows != m.cols) {
        printf("%sError: Matrix must be square.\n%s", URED, COLOR_RESET);
        free_matrix(&m);
        return 1;
    }
    matrix eigenvalues, eigenvectors;
    int max_iter = 2000;
    double tol = 1e-10;
    qr_algorithm(m, &eigenvalues, &eigenvectors, max_iter, tol);
    display_eigen(eigenvalues, eigenvectors);
    free_matrix(&eigenvalues);
    free_matrix(&eigenvectors);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int lu_decomposition_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    if (m.rows != m.cols) {
        printf("%sError: Matrix must be square.\n%s", URED, COLOR_RESET);
        free_matrix(&m);
        return 1;
    }
    matrix L, U;
    int status = lu_decomposition(m, &L, &U);
    if (status == 1) {
        printf("%sError: Matrix is singular. LU decomposition cannot be performed.\n%s", URED, COLOR_RESET);
        free_matrix(&m);
        return 1;
    }
    printf("%sLower triangular matrix L:\n%s", UGRN, COLOR_RESET);
    display_matrix(L);
    printf("%sUpper triangular matrix U:\n%s", UGRN, COLOR_RESET);
    display_matrix(U);
    free_matrix(&L);
    free_matrix(&U);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int matrix_norms_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    double f_norm = frobenius_norm(m);
    double one_n = one_norm(m);
    double inf_n = infinity_norm(m);
    printf("\n%sFrobenius norm: %f\n%s", UGRN, f_norm, COLOR_RESET);
    printf("%sOne-norm: %f\n%s", UGRN, one_n, COLOR_RESET);
    printf("%sInfinity norm: %f\n%s", UGRN, inf_n, COLOR_RESET);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int save_matrix_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    char filename[100];
    printf("%sEnter file name: %s", UBLU, COLOR_RESET);
    scanf("%s", filename);
    save_matrix_to_file(m, filename);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int load_matrix_operation(matrix *saved_matrix, int *matrix_loaded) {
    char filename[100];
    printf("%sEnter file name: %s", UBLU, COLOR_RESET);
    scanf("%s", filename);
    if (*matrix_loaded) free_matrix(saved_matrix);
    *saved_matrix = load_matrix_from_file(filename);
    if (saved_matrix->data == NULL) {
        printf("%sError: Failed to load matrix.\n%s", URED, COLOR_RESET);
        *matrix_loaded = 0;
        return 1;
    }
    *matrix_loaded = 1;
    display_matrix(*saved_matrix);
    wait_for_enter();
    return 0;
}

int use_loaded_matrix_operation(matrix *saved_matrix, int *matrix_loaded) {
    if (!*matrix_loaded) {
        printf("%sNo matrix loaded or generated. Please load or generate a matrix first.\n%s", URED, COLOR_RESET);
        wait_for_enter();
        return 1;
    }
    int op_choice;
    printf("\n%sSelect operation for loaded matrix:\n%s", UCYN, COLOR_RESET);
    printf("%s1. Add to another matrix\n%s", UYEL, COLOR_RESET);
    printf("%s2. Subtract from another matrix\n%s", UYEL, COLOR_RESET);
    printf("%s3. Multiply by another matrix\n%s", UYEL, COLOR_RESET);
    printf("%s4. Multiply by scalar\n%s", UYEL, COLOR_RESET);
    printf("%s5. Transpose\n%s", UYEL, COLOR_RESET);
    printf("%s6. Find determinant\n%s", UYEL, COLOR_RESET);
    printf("%s7. Find inverse matrix\n%s", UYEL, COLOR_RESET);
    printf("%s8. Solve system of linear equations\n%s", UYEL, COLOR_RESET);
    printf("%s9. Find rank\n%s", UYEL, COLOR_RESET);
    printf("%s11. Check matrix properties\n%s", UYEL, COLOR_RESET);
    printf("%s12. Matrix exponentiation\n%s", UYEL, COLOR_RESET);
    printf("%s13. Cholesky decomposition\n%s", UYEL, COLOR_RESET);
    printf("%s14. Eigenvalues and Eigenvector\n%s", UYEL, COLOR_RESET);
    printf("%s15. LU decomposition\n%s", UYEL, COLOR_RESET);
    printf("%s16. Matrix norms\n%s", UYEL, COLOR_RESET);
    printf("%s21. Singular Values Decomposition (SVD)\n%s", UYEL, COLOR_RESET);
    printf("%s22. Schur decomposition\n\n%s", UYEL, COLOR_RESET);
    printf("%sEnter your choice: %s", UBLU, COLOR_RESET);
    scanf("%d", &op_choice);
    getchar(); // Consume newline

    switch (op_choice) {
        case 1: { // Add to another matrix
            matrix b = input_matrix_new();
            if (saved_matrix->rows != b.rows || saved_matrix->cols != b.cols) {
                printf("%sError: Matrices must have the same dimensions for addition.\n%s", URED, COLOR_RESET);
            } else {
                matrix result = add_matrices(*saved_matrix, b);
                display_matrix(result);
                free_matrix(&result);
            }
            free_matrix(&b);
            break;
        }
        case 2: { // Subtract from another matrix
            matrix b = input_matrix_new();
            if (saved_matrix->rows != b.rows || saved_matrix->cols != b.cols) {
                printf("%sError: Matrices must have the same dimensions for subtraction.\n%s", URED, COLOR_RESET);
            } else {
                matrix result = subtract_matrices(*saved_matrix, b);
                display_matrix(result);
                free_matrix(&result);
            }
            free_matrix(&b);
            break;
        }
        case 3: { // Multiply by another matrix
            matrix b = input_matrix_new();
            if (saved_matrix->cols != b.rows) {
                printf("%sError: Number of columns in first matrix must equal number of rows in second matrix.\n%s", URED, COLOR_RESET);
            } else {
                matrix result = multiply_matrices_strassen_padded(*saved_matrix, b);
                display_matrix(result);
                free_matrix(&result);
            }
            free_matrix(&b);
            break;
        }
        case 4: { // Multiply by scalar
            double scalar = input_scalar();
            matrix result = scalar_multiply(*saved_matrix, scalar);
            display_matrix(result);
            free_matrix(&result);
            break;
        }
        case 5: { // Transpose
            matrix result = transpose_matrix(*saved_matrix);
            display_matrix(result);
            free_matrix(&result);
            break;
        }
        case 6: { // Find determinant
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Determinant is only defined for square matrices.\n%s", URED, COLOR_RESET);
            } else {
                double det = determinant(*saved_matrix);
                printf("\n%sDeterminant: %lf\n%s", UGRN, det, COLOR_RESET);
            }
            break;
        }
        case 7: { // Find inverse matrix
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Inverse is only defined for square matrices.\n%s", URED, COLOR_RESET);
            } else {
                matrix result = inverse_matrix(*saved_matrix);
                display_matrix(result);
                free_matrix(&result);
            }
            break;
        }
        case 8: { // Solve system of linear equations
            printf("%sInput vector B:\n%s", UBLU, COLOR_RESET);
            matrix b = input_matrix_new();
            if (saved_matrix->rows != b.rows || b.cols != 1) {
                printf("%sError: B must be a column vector with %d rows.\n%s", URED, saved_matrix->rows, COLOR_RESET);
            } else {
                matrix x = solve_system(*saved_matrix, b);
                display_matrix(x);
                free_matrix(&x);
            }
            free_matrix(&b);
            break;
        }
        case 9: { // Find rank
            int r = rank(*saved_matrix);
            printf("\n%sRank: %d\n%s", UGRN, r, COLOR_RESET);
            break;
        }
        case 11: { // Check matrix properties
            printf("\n%sMatrix properties:\n%s", UGRN, COLOR_RESET);
            printf("%sDiagonal: %s\n%s", UGRN, is_diagonal(*saved_matrix) ? "Yes" : "No", COLOR_RESET);
            printf("%sSymmetric: %s\n%s", UGRN, is_symmetric(*saved_matrix) ? "Yes" : "No", COLOR_RESET);
            printf("%sUpper triangular: %s\n%s", UGRN, is_upper_triangular(*saved_matrix) ? "Yes" : "No", COLOR_RESET);
            printf("%sLower triangular: %s\n%s", UGRN, is_lower_triangular(*saved_matrix) ? "Yes" : "No", COLOR_RESET);
            printf("%sIdentity: %s\n%s", UGRN, is_identity(*saved_matrix) ? "Yes" : "No", COLOR_RESET);
            break;
        }
        case 12: { // Matrix exponentiation
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Matrix must be square for exponentiation.\n%s", URED, COLOR_RESET);
            } else {
                int exponent;
                printf("%sEnter exponent: %s", UBLU, COLOR_RESET);
                scanf("%d", &exponent);
                matrix result = matrix_power(*saved_matrix, exponent);
                display_matrix(result);
                free_matrix(&result);
            }
            break;
        }
        case 13: { // Cholesky decomposition
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Matrix must be square for Cholesky decomposition.\n%s", URED, COLOR_RESET);
            } else if (!is_symmetric(*saved_matrix)) {
                printf("%sError: Matrix must be symmetric.\n%s", URED, COLOR_RESET);
            } else {
                matrix L = cholesky_decomposition(*saved_matrix);
                printf("%sLower triangular matrix L:\n%s", UGRN, COLOR_RESET);
                display_matrix(L);
                free_matrix(&L);
            }
            break;
        }
        case 14: { // Eigenvalues and Eigenvector
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Matrix must be square.\n%s", URED, COLOR_RESET);
            } else {
                matrix eigenvalues, eigenvectors;
                int max_iter = 2000;
                double tol = 1e-10;
                qr_algorithm(*saved_matrix, &eigenvalues, &eigenvectors, max_iter, tol);
                display_eigen(eigenvalues, eigenvectors);
                free_matrix(&eigenvalues);
                free_matrix(&eigenvectors);
            }
            break;
        }
        case 15: { // LU decomposition
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Matrix must be square.\n%s", URED, COLOR_RESET);
            } else {
                matrix L, U;
                int status = lu_decomposition(*saved_matrix, &L, &U);
                if (status == 1) {
                    printf("%sError: Matrix is singular. LU decomposition cannot be performed.\n%s", URED, COLOR_RESET);
                } else {
                    printf("%sLower triangular matrix L:\n%s", UGRN, COLOR_RESET);
                    display_matrix(L);
                    printf("%sUpper triangular matrix U:\n%s", UGRN, COLOR_RESET);
                    display_matrix(U);
                    free_matrix(&L);
                    free_matrix(&U);
                }
            }
            break;
        }
        case 16: { // Matrix norms
            double f_norm = frobenius_norm(*saved_matrix);
            double one_n = one_norm(*saved_matrix);
            double inf_n = infinity_norm(*saved_matrix);
            printf("\n%sFrobenius norm: %f\n%s", UGRN, f_norm, COLOR_RESET);
            printf("%sOne-norm: %f\n%s", UGRN, one_n, COLOR_RESET);
            printf("%sInfinity norm: %f\n%s", UGRN, inf_n, COLOR_RESET);
            break;
        }
        case 21: { // SVD
            printf("\n%sSingular value decomposition (SVD):%s\n", UGRN, COLOR_RESET);
            matrix U, Sigma, V;
            svd(*saved_matrix, &U, &Sigma, &V);
            printf("%sMatrix U:%s\n", UGRN, COLOR_RESET);
            display_matrix(U);
            printf("%sMatrix Sigma:%s\n", UGRN, COLOR_RESET);
            display_matrix(Sigma);
            printf("%sMatrix V:%s\n", UGRN, COLOR_RESET);
            display_matrix(V);
            free_matrix(&U);
            free_matrix(&Sigma);
            free_matrix(&V);
            break;
        }
        case 22: { // Schur decomposition
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Matrix must be square.\n%s", URED, COLOR_RESET);
            } else {
                matrix Q, T;
                int max_iter = 1000;
                double tol = 1e-8;
                schur_decomposition(*saved_matrix, &Q, &T, max_iter, tol);
                printf("%sOrthogonal matrix Q:\n%s", UGRN, COLOR_RESET);
                display_matrix(Q);
                printf("%sQuasi-triangular matrix T:\n%s", UGRN, COLOR_RESET);
                display_matrix(T);
                free_matrix(&Q);
                free_matrix(&T);
            }
            break;
        }
        default:
            printf("%sInvalid operation choice.\n%s", URED, COLOR_RESET);
    }
    wait_for_enter();
    return 0;
}

int save_random_matrix_operation(matrix *saved_matrix, int *matrix_loaded) {
    if (!*matrix_loaded) {
        printf("%sNo random matrix generated. Please generate a matrix first.\n%s", URED, COLOR_RESET);
        wait_for_enter();
        return 1;
    }
    char filename[100];
    printf("%sEnter file name: %s", UBLU, COLOR_RESET);
    scanf("%s", filename);
    save_matrix_to_file(*saved_matrix, filename);
    wait_for_enter();
    return 0;
}

int svd_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    printf("\n%sSingular value decomposition (SVD):%s\n", UGRN, COLOR_RESET);
    matrix U, Sigma, V;
    svd(m, &U, &Sigma, &V);
    printf("%sMatrix U:%s\n", UGRN, COLOR_RESET);
    display_matrix(U);
    printf("%sMatrix Sigma:%s\n", UGRN, COLOR_RESET);
    display_matrix(Sigma);
    printf("%sMatrix V:%s\n", UGRN, COLOR_RESET);
    display_matrix(V);
    free_matrix(&U);
    free_matrix(&Sigma);
    free_matrix(&V);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int schur_decomposition_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    if (m.rows != m.cols) {
        printf("%sError: Matrix must be square.\n%s", URED, COLOR_RESET);
        free_matrix(&m);
        return 1;
    }
    matrix Q, T;
    int max_iter = 1000;
    double tol = 1e-8;
    schur_decomposition(m, &Q, &T, max_iter, tol);
    printf("%sOrthogonal matrix Q:\n%s", UGRN, COLOR_RESET);
    display_matrix(Q);
    printf("%sQuasi-triangular matrix T:\n%s", UGRN, COLOR_RESET);
    display_matrix(T);
    free_matrix(&Q);
    free_matrix(&T);
    free_matrix(&m);
    wait_for_enter();
    return 0;
}

int exit_operation(matrix *saved_matrix, int *matrix_loaded) {
    printf("\nExit...\n");
    exit(0);
    return 0;
}

/* Array of function pointers for operations */
typedef int (*operation_func)(matrix *, int *);

operation_func operations[] = {
    NULL, // Index 0 is not used
    add_matrices_operation,
    subtract_matrices_operation,
    multiply_matrices_operation,
    scalar_multiply_operation,
    transpose_operation,
    determinant_operation,
    inverse_operation,
    solve_system_operation,
    rank_operation,
    generate_random_matrix_operation,
    check_properties_operation,
    matrix_power_operation,
    cholesky_decomposition_operation,
    eigenvalues_operation,
    lu_decomposition_operation,
    matrix_norms_operation,
    save_matrix_operation,
    load_matrix_operation,
    use_loaded_matrix_operation,
    save_random_matrix_operation,
    svd_operation,
    schur_decomposition_operation,
    exit_operation
};

/* Main function - manages the program's logic */
int main() {
    srand(time(NULL));      // Initialize the random number generator
    int choice;
    matrix saved_matrix = {0, 0, NULL};    // Initialize saved_matrix
    int matrix_loaded = 0;  // Flag for loaded matrix

    do {
        show_menu();
        choice = get_user_choice();
        if (choice >= 1 && choice <= 23) {
            int error = operations[choice](&saved_matrix, &matrix_loaded);
            if (error) {
                printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
            }
        } else {
            printf("\n%sWrong choice. Please enter a number between 1 and 23.\n%s", URED, COLOR_RESET);
        }
    } while (choice != 23);

    if (matrix_loaded && saved_matrix.data != NULL) {
        free_matrix(&saved_matrix);
    }

    return 0;
}