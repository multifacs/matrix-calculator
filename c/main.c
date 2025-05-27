/* 
* Matrix Calculator is designed to perform matrix operations.
* It supports basic actions and complex computations,
* such as determinant, inverse matrix, decompositions and eigenvalues. 
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

/* 
* Display the main menu with available operations.
* Improves perception and makes more user-friendly interface. 
*/
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
    printf("%s21. Exit\n\n%s", UYEL, COLOR_RESET);
}

/* 
* Gets the user's choice from menu.
* Used infinite loop so ensures the program doesn't terminate due to incorrect input.
*/
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
        if (*endptr == '\0' && choice >= 1 && choice <= 21) {
            return (int)choice;
        } else {
            printf("%sInvalid input. Please enter a number between 1 and 21.\n%s", URED, COLOR_RESET);
        }
    }
}

// Requests a positive integer from the user.
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

/* 
* Collects parameters for generating a random matrix.
* Min_val and max_val checked to avoid logical errors in generation. 
*/
void input_random_matrix_params(int *rows, int *cols, double *min_val, double *max_val) {
    *rows = input_positive_integer("\nEnter number of rows: ");
    *cols = input_positive_integer("Enter number of cols: ");
    printf("%sEnter minimum value for elements: %s", UBLU, COLOR_RESET);
    while (scanf("%lf", min_val) != 1) {
        while (getchar() != '\n');
        printf("Invalid input. Please enter a number.\n%s", URED, COLOR_RESET);
        printf("Enter minimum value for elements: %s", UBLU, COLOR_RESET);
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

/* 
 * Creates a new matrix by requesting dimensions and elements.
 * Separated dimension and element inputs for simplification of code reuse.
 */
matrix input_matrix_new() {
    int rows = input_positive_integer("\nEnter number of rows: ");
    int cols = input_positive_integer("Enter number of columns: ");
    matrix m = create_matrix(rows, cols);
    input_matrix(&m);
    edit_matrix(&m);
    return m;
}

/* 
 * Displays the matrix on the screen using the print_matrix function.
 * Enhances modularity and code readability.
 */
void display_matrix(matrix m) {
    print_matrix(m);
}

// Requests a scalar value from the user.
double input_scalar() {
    double scalar;
    printf("%sInput scalar: %s", UBLU, COLOR_RESET);
    scanf("%lf", &scalar);
    return scalar;
}

/* 
 * Waits for Enter to be pressed before continuing.
 * User can review the results before the next operation.
 */
void wait_for_enter() {
    printf("%sPress Enter to continue...\n%s", UBLU, COLOR_RESET);
    while (getchar() != '\n');
}

// Displays eigenvalues and eigenvectors
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

// Save matrix into txt file
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

// Load matrix from txt file
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
            if (fscanf(file, "%lf", &m.data[i][j]) != 1 ) {
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

/* 
 * Function to select and perform operations with the loaded matrix
 */
void use_loaded_matrix(matrix loaded_matrix) {
    int op_choice;
    printf("\n%sSelect operation for loaded matrix:\n%s", UCYN, COLOR_RESET);
    printf("%s1. Add to another matrix\n%s", UYEL, COLOR_RESET);
    printf("%s2. Subtract from another matrix\n%s", UYEL, COLOR_RESET);
    printf("%s3. Multiply by another matrix\n%s", UYEL, COLOR_RESET);
    printf("%s4. Multiply by scalar\n%s", UYEL, COLOR_RESET);
    printf("%s5. Transpose\n%s", UYEL, COLOR_RESET);
    printf("%s6. Find determinant\n%s", UYEL, COLOR_RESET);
    printf("%s7. Find inverse matrix\n%s", UYEL, COLOR_RESET);
    printf("%s8. Find rank\n%s", UYEL, COLOR_RESET);
    printf("%s9. Check matrix properties\n%s", UYEL, COLOR_RESET);
    printf("%s10. Matrix exponentiation\n%s", UYEL, COLOR_RESET);
    printf("%s11. Cholesky decomposition\n%s", UYEL, COLOR_RESET);
    printf("%s12. Eigenvalues and Eigenvector\n%s", UYEL, COLOR_RESET);
    printf("%s13. LU decomposition\n%s", UYEL, COLOR_RESET);
    printf("%s14. Matrix norms\n%s", UYEL, COLOR_RESET);
    printf("%sEnter your choice: %s", UBLU, COLOR_RESET);
    scanf("%d", &op_choice);
    getchar(); // Consume newline

    switch (op_choice) {
        case 1: { // Add to another matrix
            matrix b = input_matrix_new();
            if (loaded_matrix.rows != b.rows || loaded_matrix.cols != b.cols) {
                printf("%sError: Matrices must have the same dimensions for addition.\n%s", URED, COLOR_RESET);
            } else {
                matrix result = add_matrices(loaded_matrix, b);
                display_matrix(result);
                free_matrix(&result);
            }
            free_matrix(&b);
            break;
        }
        case 2: { // Subtract from another matrix
            matrix b = input_matrix_new();
            if (loaded_matrix.rows != b.rows || loaded_matrix.cols != b.cols) {
                printf("%sError: Matrices must have the same dimensions for subtraction.\n%s", URED, COLOR_RESET);
            } else {
                matrix result = subtract_matrices(loaded_matrix, b);
                display_matrix(result);
                free_matrix(&result);
            }
            free_matrix(&b);
            break;
        }
        case 3: { // Multiply by another matrix
            matrix b = input_matrix_new();
            if (loaded_matrix.cols != b.rows) {
                printf("%sError: Number of columns in first matrix must equal number of rows in second matrix.\n%s", URED, COLOR_RESET);
            } else {
                matrix result = multiply_matrices(loaded_matrix, b);
                display_matrix(result);
                free_matrix(&result);
            }
            free_matrix(&b);
            break;
        }
        case 4: { // Multiply by scalar
            double scalar = input_scalar();
            matrix result = scalar_multiply(loaded_matrix, scalar);
            display_matrix(result);
            free_matrix(&result);
            break;
        }
        case 5: { // Transpose
            matrix result = transpose_matrix(loaded_matrix);
            display_matrix(result);
            free_matrix(&result);
            break;
        }
        case 6: { // Find determinant
            if (loaded_matrix.rows != loaded_matrix.cols) {
                printf("%sError: Determinant is only defined for square matrices.\n%s", URED, COLOR_RESET);
            } else {
                double det = determinant(loaded_matrix);
                printf("\n%sDeterminant: %lf\n%s", UGRN, det, COLOR_RESET);
            }
            break;
        }
        case 7: { // Find inverse matrix
            if (loaded_matrix.rows != loaded_matrix.cols) {
                printf("%sError: Inverse is only defined for square matrices.\n%s", URED, COLOR_RESET);
            } else {
                matrix result = inverse_matrix(loaded_matrix);
                display_matrix(result);
                free_matrix(&result);
            }
            break;
        }
        case 8: { // Find rank
            int r = rank(loaded_matrix);
            printf("\n%sRank: %d\n%s", UGRN, r, COLOR_RESET);
            break;
        }
        case 9: { // Check matrix properties
            printf("\n%sMatrix properties:\n%s", UGRN, COLOR_RESET);
            printf("%sDiagonal: %s\n%s", UGRN, is_diagonal(loaded_matrix) ? "Yes" : "No", COLOR_RESET);
            printf("%sSymmetric: %s\n%s", UGRN, is_symmetric(loaded_matrix) ? "Yes" : "No", COLOR_RESET);
            printf("%sUpper triangular: %s\n%s", UGRN, is_upper_triangular(loaded_matrix) ? "Yes" : "No", COLOR_RESET);
            printf("%sLower triangular: %s\n%s", UGRN, is_lower_triangular(loaded_matrix) ? "Yes" : "No", COLOR_RESET);
            printf("%sIdentity: %s\n%s", UGRN, is_identity(loaded_matrix) ? "Yes" : "No", COLOR_RESET);
            break;
        }
        case 10: { // Matrix exponentiation
            if (loaded_matrix.rows != loaded_matrix.cols) {
                printf("%sError: Matrix must be square for exponentiation.\n%s", URED, COLOR_RESET);
            } else {
                int exponent;
                printf("%sEnter exponent: %s", UBLU, COLOR_RESET);
                scanf("%d", &exponent);
                matrix result = matrix_power(loaded_matrix, exponent);
                display_matrix(result);
                free_matrix(&result);
            }
            break;
        }
        case 11: { // Cholesky decomposition
            if (loaded_matrix.rows != loaded_matrix.cols) {
                printf("%sError: Matrix must be square for Cholesky decomposition.\n%s", URED, COLOR_RESET);
            } else if (!is_symmetric(loaded_matrix)) {
                printf("%sError: Matrix must be symmetric.\n%s", URED, COLOR_RESET);
            } else {
                matrix L = cholesky_decomposition(loaded_matrix);
                printf("%sLower triangular matrix L:\n%s", UGRN, COLOR_RESET);
                display_matrix(L);
                free_matrix(&L);
            }
            break;
        }
        case 12: { // Eigenvalues and Eigenvector
            if (loaded_matrix.rows != loaded_matrix.cols) {
                    printf("%sError: matrix must be square.\n%s", URED, COLOR_RESET);
                } else {
                    matrix eigenvalues, eigenvectors;
                    int max_iter = 1000;
                    double tol = 1e-6;
                    
                    qr_algorithm(loaded_matrix, &eigenvalues, &eigenvectors, max_iter, tol);
                    display_eigen(eigenvalues, eigenvectors);

                    free_matrix(&eigenvalues);
                    free_matrix(&eigenvectors);
                }
                break;
        }
        case 13: { // LU decomposition
            if (loaded_matrix.rows != loaded_matrix.cols) {
                printf("%sError: Matrix must be square.\n%s", URED, COLOR_RESET);
            } else {
                matrix L, U;
                lu_decomposition(loaded_matrix, &L, &U);
                printf("%sLower triangular matrix L:\n%s", UGRN, COLOR_RESET);
                display_matrix(L);
                printf("%sUpper triangular matrix U:\n%s", UGRN, COLOR_RESET);
                display_matrix(U);
                free_matrix(&L);
                free_matrix(&U);
            }
            break;
        }
        case 14: { // Matrix norms
            double f_norm = frobenius_norm(loaded_matrix);
            double one_n = one_norm(loaded_matrix);
            double inf_n = infinity_norm(loaded_matrix);
            printf("\n%sFrobenius norm: %f\n%s", UGRN, f_norm, COLOR_RESET);
            printf("%sOne-norm: %f\n%s", UGRN, one_n, COLOR_RESET);
            printf("%sInfinity norm: %f\n%s", UGRN, inf_n, COLOR_RESET);
            break;
        }
        default:
            printf("%sInvalid operation choice.\n%s", URED, COLOR_RESET);
    }
}

/* 
 * Main function - manages the program's logic.
 * Provides clear separation of operations.
 */
int main() {
    srand(time(NULL));      // Initialize the random number generator for matrix creation
    int choice;
    matrix saved_matrix;    // Variable for keeping loaded or generated matrix
    int matrix_loaded = 0;  // Flag for loaded matrix

    do {
        show_menu();
        choice = get_user_choice();
        switch(choice) {
            case 1: {
                printf("\n%sAdd two matrices:\n%s", UGRN, COLOR_RESET);
                matrix a = input_matrix_new();
                matrix b = input_matrix_new();
                matrix result = add_matrices(a, b);
                display_matrix(result);
                free_matrix(&a);
                free_matrix(&b);
                free_matrix(&result);
                wait_for_enter();
                break;
            }
            case 2: {
                printf("\n%sSubtract two matrices:\n%s", UGRN, COLOR_RESET);
                matrix a = input_matrix_new();
                matrix b = input_matrix_new();
                matrix result = subtract_matrices(a, b);
                display_matrix(result);
                free_matrix(&a);
                free_matrix(&b);
                free_matrix(&result);
                wait_for_enter();
                break;
            }
            case 3: {
                printf("\n%sMultiply two matrices:\n%s", UGRN, COLOR_RESET);
                matrix a = input_matrix_new();
                matrix b = input_matrix_new();

                // Check if Strassen's algorithm if usable
                int is_square = (a.rows == a.cols) && (b.rows == b.cols) && (a.rows == b.rows);
                int is_power_of_two_size = is_power_of_two(a.rows);

                if (is_square && is_power_of_two_size) {    // Use Strassen's if matrices are square and size of matrices are powers of 2
                    matrix result = multiply_matrices_strassen(a, b);
                    display_matrix(result);
                    free_matrix(&result);
                } else if (a.cols == b.rows) {      // Use standart multiplication algorithm
                    matrix result = multiply_matrices(a, b);
                    display_matrix(result);
                    free_matrix(&result);
                } else {        // Matrices are incompatable for multiplication
                    printf("%sError: matrix sizes are incompatable for multiplication. \n%s", URED, COLOR_RESET);
                }
                
                free_matrix(&a);
                free_matrix(&b);

                wait_for_enter();

                break;
            }
            case 4: {
                printf("\n%sMultiply matrix by a scalar:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                double scalar = input_scalar();
                matrix result = scalar_multiply(m, scalar);
                display_matrix(result);
                free_matrix(&m);
                free_matrix(&result);
                wait_for_enter();
                break;
            }
            case 5: {
                printf("\n%sMatrix transposition:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                matrix result = transpose_matrix(m);
                display_matrix(result);
                free_matrix(&m);
                free_matrix(&result);
                wait_for_enter();
                break;
            }
            case 6: {
                printf("\n%sFind determinant:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("%sError: determinant is valid only for square matrices.\n%s", URED, COLOR_RESET);
                } else {
                    double det = determinant(m);
                    printf("\n%sDeterminant: %lf\n%s", UGRN, det, COLOR_RESET);
                }
                free_matrix(&m);
                wait_for_enter();
                break;
            }
            case 7: {
                printf("\n%sFind inverse matrix:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("%sError: inverse matrix is valid only for square matrices.\n%s", URED, COLOR_RESET);
                } else {
                    matrix result = inverse_matrix(m);
                    display_matrix(result);
                    free_matrix(&result);
                }
                free_matrix(&m);
                wait_for_enter();
                break;
            }
            case 8: {
                printf("\n%sSolve SLE (Ax = B):\n%s", UGRN, COLOR_RESET);
                printf("%sInput matrix A:\n%s", UBLU, COLOR_RESET);
                matrix A = input_matrix_new();
                printf("%sInput vector B:\n%s", UBLU, COLOR_RESET);
                matrix B = input_matrix_new();
                if (A.rows != B.rows || B.cols != 1) {
                    printf("%sError: B must be a column vector with %d rows.\n%s", URED, A.rows, COLOR_RESET);
                } else {
                    matrix x = solve_system(A, B);
                    display_matrix(x);
                    free_matrix(&x);
                }
                free_matrix(&A);
                free_matrix(&B);
                wait_for_enter();
                break;
            }
            case 9: {
                printf("\n%sFind rank of a matrix:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                int r = rank(m);
                printf("\n%sRank: %d\n%s", UGRN, r, COLOR_RESET);
                free_matrix(&m);
                wait_for_enter();
                break;
            }
            case 10: {
                printf("\n%sGenerate random matrix:\n%s", UGRN, COLOR_RESET);
                int rows, cols;
                double min_val, max_val;
                input_random_matrix_params(&rows, &cols, &min_val, &max_val);
                if (matrix_loaded) free_matrix(&saved_matrix);
                saved_matrix = generate_random_matrix(rows, cols, min_val, max_val);
                matrix_loaded = 1;
                printf("%sGenerated random matrix:\n%s", UGRN, COLOR_RESET);
                display_matrix(saved_matrix);
                wait_for_enter();
                break;
            }
            case 11: {
                printf("\n%sCheck matrix properties:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                printf("\n%sMatrix properties:\n%s", UGRN, COLOR_RESET);
                printf("%sDiagonal: %s\n%s", UGRN, is_diagonal(m) ? "Yes" : "No", COLOR_RESET);
                printf("%sSymmetric: %s\n%s", UGRN, is_symmetric(m) ? "Yes" : "No", COLOR_RESET);
                printf("%sUpper triangular: %s\n%s", UGRN, is_upper_triangular(m) ? "Yes" : "No", COLOR_RESET);
                printf("%sLower triangular: %s\n%s", UGRN, is_lower_triangular(m) ? "Yes" : "No", COLOR_RESET);
                printf("%sIdentity: %s\n%s", UGRN, is_identity(m) ? "Yes" : "No", COLOR_RESET);
                free_matrix(&m);
                wait_for_enter();
                break;
            }
            case 12: {
                printf("\n%sMatrix exponentiation:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("%sError: matrix must be square.\n%s", URED, COLOR_RESET);
                } else {
                    int exponent;
                    printf("%sEnter exponent: %s", UBLU, COLOR_RESET);
                    scanf("%d", &exponent);
                    matrix result = matrix_power(m, exponent);
                    display_matrix(result);
                    free_matrix(&result);
                }
                free_matrix(&m);
                wait_for_enter();
                break;
            }
            case 13: {
                printf("\n%sCholesky decomposition:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("%sError: matrix must be square.\n%s", URED, COLOR_RESET);
                } else if (!is_symmetric(m)) {
                    printf("%sError: matrix must be symmetric.\n%s", URED, COLOR_RESET);
                } else {
                    matrix L = cholesky_decomposition(m);
                    printf("%sLower triangular matrix L:\n%s", UGRN, COLOR_RESET);
                    display_matrix(L);
                    free_matrix(&L);
                }
                free_matrix(&m);
                wait_for_enter();
                break;
            }
            case 14: {
                printf("\n%sEigenvalues and Eigenvectors:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("%sError: matrix must be square.\n%s", URED, COLOR_RESET);
                } else {
                    matrix eigenvalues, eigenvectors;
                    int max_iter = 1000;
                    double tol = 1e-6;
                    
                    qr_algorithm(m, &eigenvalues, &eigenvectors, max_iter, tol);
                    display_eigen(eigenvalues, eigenvectors);

                    free_matrix(&eigenvalues);
                    free_matrix(&eigenvectors);
                }
                free_matrix(&m);

                wait_for_enter();
                break;
            }
            case 15: {
                printf("\n%sLU decomposition:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("%sError: matrix must be square\n%s", URED, COLOR_RESET);
                } else {
                    matrix L, U;
                    lu_decomposition(m, &L, &U);
                    printf("%sLower triangular matrix L:\n%s", UGRN, COLOR_RESET);
                    display_matrix(L);
                    printf("%sUpper triangular matrix U:\n%s", UGRN, COLOR_RESET);
                    display_matrix(U);
                    free_matrix(&L);
                    free_matrix(&U);
                }
                free_matrix(&m);
                wait_for_enter();
                break;
            }
            case 16: {
                printf("\n%sMatrix norms:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                double f_norm = frobenius_norm(m);
                double one_n = one_norm(m);
                double inf_n = infinity_norm(m);
                printf("\n%sFrobenius norm: %f\n%s", UGRN, f_norm, COLOR_RESET);
                printf("%sOne-norm: %f\n%s", UGRN, one_n, COLOR_RESET);
                printf("%sInfinity norm: %f\n%s", UGRN, inf_n, COLOR_RESET);
                free_matrix(&m);
                wait_for_enter();
                break;
            }
            case 17: {
                printf("\n%sSave matrix in file:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                char filename[100];
                printf("%sEnter file name: %s", UBLU, COLOR_RESET);
                scanf("%s", filename);
                save_matrix_to_file(m, filename);
                free_matrix(&m);
                wait_for_enter();
                break;
            }
            case 18: {
                printf("\n%sLoading matrix from file:\n%s", UGRN, COLOR_RESET);
                char filename[100];
                printf("%sEnter file name: %s", UBLU, COLOR_RESET);
                scanf("%s", filename);
                if (matrix_loaded) free_matrix(&saved_matrix);
                saved_matrix = load_matrix_from_file(filename);
                if (saved_matrix.data != NULL) {
                    matrix_loaded = 1;
                    display_matrix(saved_matrix);
                }
                wait_for_enter();
                break;
            }
            case 19: {
                if (!matrix_loaded) {
                    printf("%sNo matrix loaded or generated. Please load or generate a matrix first.\n%s", URED, COLOR_RESET);
                    wait_for_enter();
                    break;
                }
                use_loaded_matrix(saved_matrix);
                wait_for_enter();
                break;
            }
            case 20: {
                if (!matrix_loaded) {
                    printf("%sNo random matrix generated. Please generate a matrix first.\n%s", URED, COLOR_RESET);
                    wait_for_enter();
                    break;
                }
                printf("\n%sSave random matrix to file:\n%s", UGRN, COLOR_RESET);
                char filename[100];
                printf("%sEnter file name: %s", UBLU, COLOR_RESET);
                scanf("%s", filename);
                save_matrix_to_file(saved_matrix, filename);
                wait_for_enter();
                break;
            }
            case 21:
                printf("\nExit...\n");
                break;
            default:
                printf("\n%sWrong choice. Please enter a number between 1 and 21.\n%s", URED, COLOR_RESET);
        }
    } while (choice != 21);

    if (matrix_loaded && saved_matrix.data != NULL) {
    free_matrix(&saved_matrix);
    }

    return 0;
}