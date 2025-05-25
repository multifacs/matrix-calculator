#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <time.h>
#include "matrix.h"

// TODO matrix format
// TODO errors log
// TODO change elements of defined matrix
// TODO help
// TODO Web interface (WebAssebly?)
// TODO save and load matricies
// FIXME better matrix input

#define URED            "\e[4;31m"      // Errors
#define UGRN            "\e[4;32m"      // Results
#define UYEL            "\e[4;33m"      // Information/warnings
#define UBLU            "\e[4;34m"      // Inputs
#define UCYN            "\e[4;36m"      // Menus
#define COLOR_RESET     "\e[0m"         // Color reeset


void show_menu() {
    printf("%sMATRIX CALCULATOR\n%s", UCYN, COLOR_RESET);
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
    printf("%s17. Exit\n%s", UYEL, COLOR_RESET);
}

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
        if (*endptr == '\0' && choice >= 1 && choice <= 17) {
            return (int)choice;
        } else {
            printf("%sInvalid input. Please enter a number between 1 and 17.\n%s", URED, COLOR_RESET);
        }
    }
}

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

void input_random_matrix_params(int *rows, int *cols, double *min_val, double *max_val) {
    *rows = input_positive_integer("Enter number of rows: ");
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

matrix input_matrix_new() {
    int rows = input_positive_integer("Enter number of rows: ");
    int cols = input_positive_integer("Enter number of columns: ");
    matrix m = create_matrix(rows, cols);
    input_matrix(&m);
    return m;
}

void display_matrix(matrix m) {
    print_matrix(m);
}

int input_scalar() {
    double scalar;
    printf("%sInput scalar: %s", UBLU, COLOR_RESET);
    scanf("%lf", &scalar);
    return scalar;
}

void wait_for_enter() {
    printf("%sPress Enter to continue...\n%s", UBLU, COLOR_RESET);
    while (getchar() != '\n');
}

int main() {
    srand(time(NULL));
    int choice;
    do {
        show_menu();
        choice = get_user_choice();
        switch(choice) {
            case 1: {
                printf("%sAdd two matrices:\n%s", UGRN, COLOR_RESET);

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
                printf("%sSubtract two matrices:\n%s", UGRN, COLOR_RESET);

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
                printf("%sMultiply two matrices:\n%s", UGRN, COLOR_RESET);

                matrix a = input_matrix_new();
                matrix b = input_matrix_new();

                matrix result = multiply_matrices(a, b);

                display_matrix(result);

                free_matrix(&a);
                free_matrix(&b);
                free_matrix(&result);

                wait_for_enter();
                break;
            }

            case 4: {
                printf("%sMultiply matrix by a scalar:\n%s", UGRN, COLOR_RESET);

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
                printf("%sMatrix transposition:\n%s", UGRN, COLOR_RESET);

                matrix m = input_matrix_new();
                matrix result = transpose_matrix(m);

                display_matrix(result);

                free_matrix(&m);
                free_matrix(&result);

                wait_for_enter();
                break;
            }

            case 6: {
                printf("%sFind determinant:\n%s", UGRN, COLOR_RESET);

                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("%sError: determinant is valid only for square matrices.\n%s", URED, COLOR_RESET);
                } 
                else {
                    double det = determinant(m);
                    printf("%sDeterminant: %lf\n%s", UGRN, det, COLOR_RESET);
                }

                free_matrix(&m);

                wait_for_enter();
                break;
            }
            
            case 7: {
                printf("%sFind inverse matrix:\n%s", UGRN, COLOR_RESET);

                matrix m = input_matrix_new();

                if (m.rows != m.cols) {
                    printf("%sError: inverse matrix is valid only for square matrices.\n%s", URED, COLOR_RESET);
                } 
                else {
                    matrix result = inverse_matrix(m);
                    display_matrix(result);
                    free_matrix(&result);
                }

                free_matrix(&m);

                wait_for_enter();
                break;
            }

            case 8: {
                printf("%sSolve SLE (Ax = B):\n%s", UGRN, COLOR_RESET);

                printf("%sInput matrix A:\n%s", UBLU, COLOR_RESET);
                matrix A = input_matrix_new();

                printf("%sInput vector B:\n%s", UBLU, COLOR_RESET);
                matrix B = input_matrix_new();

                if (A.rows != B.rows || B.cols != 1) {
                    printf("%sError: B must be a column vector with %d rows.\n%s", URED, A.rows, COLOR_RESET);
                }
                else {
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
                printf("%sFind rank of a matrix:\n%s", UGRN, COLOR_RESET);

                matrix m = input_matrix_new();
                int r = rank(m);

                printf("%sRank: %d\n%s", UGRN, r, COLOR_RESET);

                free_matrix(&m);

                wait_for_enter();
                break;
            }

            case 10: {
                printf("%sGenerate random matrix:\n%s", UGRN, COLOR_RESET);

                int rows, cols;
                double min_val, max_val;

                input_random_matrix_params(&rows, &cols, &min_val, &max_val);
                matrix random_matrix = generate_random_matrix(rows, cols, min_val, max_val);

                printf("%sGenerate random matrix:\n%s", UGRN, COLOR_RESET);

                print_matrix(random_matrix);

                free_matrix(&random_matrix);

                wait_for_enter();
                break;
            }

            case 11: {
                printf("%sCheck matrix properites:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                
                printf("%sMatrix properties:\n%s", UGRN, COLOR_RESET);
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
                printf("%sMatrix exponentiation:\n%s", UGRN, COLOR_RESET);
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
                printf("%sCholesky decomposition:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("%sError: matris must be square.\n%s", URED, COLOR_RESET);
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
                printf("%sEigenvalues and Eigenvectors:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("%sError: matrix must be square.\n%s", URED, COLOR_RESET);
                } else {
                    double eigenvalue;
                    matrix eigenvector;
                    int max_iter = 1000;
                    double tol = 1e-6;
                    power_method(m, &eigenvalue, &eigenvector, max_iter, tol);
                    printf("%sEigenvalue: %f\n%s", UGRN, eigenvalue, COLOR_RESET);
                    printf("%sEigenvector:\n%s", UGRN, COLOR_RESET);
                    display_matrix(eigenvector);
                    free_matrix(&eigenvector);
                }

                free_matrix(&m);
                wait_for_enter();
                break;
            }

            case 15: {
                printf("%sLU decomposition:\n%s", UGRN, COLOR_RESET);
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
                printf("%sMatrix norms:\n%s", UGRN, COLOR_RESET);
                matrix m = input_matrix_new();
                double f_norm = frobenius_norm(m);
                double one_n = one_norm(m);
                double inf_n = infinity_norm(m);
                printf("%sFrobenius norm: %f\n%s", UGRN, f_norm, COLOR_RESET);
                printf("%sOne-norm: %f\n%s", UGRN, one_n, COLOR_RESET);
                printf("%sInfinity norm: %f\n%s", UGRN, inf_n, COLOR_RESET);

                free_matrix(&m);
                wait_for_enter();
                break;
            }
        
            case 17: 
                printf("Exit...\n");
                break;
            default:
                printf("%sWrong choice. Please enter a number between 1 and 17.\n%s", URED, COLOR_RESET);
        }
    } while (choice != 17);
    return 0;
}