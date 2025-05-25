#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <time.h>
#include "matrix.h"

// TODO Add beter interface (colors?, GUI?)
// TODO matrix format
// TODO errors log
// TODO change elements of defined matrix
// TODO help
// TODO Web interface (WebAssebly?)
// TODO save and load matricies
// FIXME better matrix input



void show_menu() {
    printf("MATRIX CALCULATOR\n");
    printf("1. Add two matrices\n");
    printf("2. Subtract two matrices\n");
    printf("3. Multiply two matrices\n");
    printf("4. Multiply matrix by a scalar\n");
    printf("5. Transpose matrix\n");
    printf("6. Find determinant\n");
    printf("7. Find inverse matrix\n");
    printf("8. Solve system of linear equations\n");
    printf("9. Find rank of a matrix\n");
    printf("10. Generate random matrix\n");
    printf("11. Check matrix properties\n");
    printf("12. Matrix exponentiation\n");
    printf("13. Cholesky decomposition\n");
    printf("14. Eigenvalues and Eigenvector\n");
    printf("15. LU decomposition\n");
    printf("16. Matrix norms\n");
    printf("17. Exit\n");
}

int get_user_choice() {
    char input[100];
    while (1) {
        printf("Enter your choice: ");
        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("End of input. Exiting.\n");
            exit(0);
        }
        input[strcspn(input, "\n")] = 0;
        char *endptr;
        long choice = strtol(input, &endptr, 10);
        if (*endptr == '\0' && choice >= 1 && choice <= 17) {
            return (int)choice;
        } else {
            printf("Invalid input. Please enter a number between 1 and 17.\n");
        }
    }
}

int input_positive_integer(const char* prompt) {
    char input[100];
    int value;
    while (1) {
        printf("%s", prompt);
        if (fgets(input, sizeof(input), stdin) == NULL) {
            printf("Error reading input.\n");
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
            printf("Invalid input. Please enter a positive integer (no decimals or letters).\n");
        }
    }
}

void input_random_matrix_params(int *rows, int *cols, double *min_val, double *max_val) {
    *rows = input_positive_integer("Enter number of rows: ");
    *cols = input_positive_integer("Enter number of cols: ");
    printf("Enter minimum value for elements: ");
    while (scanf("%lf", min_val) != 1) {
        while (getchar() != '\n');
        printf("Invalid input. Please enter a number.\n");
        printf("Enter minimum value for elements: ");
    }
    while (getchar() != '\n');
    printf("Enter maximum value for elements: ");
    while (scanf("%lf", max_val) != 1 || *max_val < *min_val) {
        while (getchar() != '\n');
        if (*max_val < *min_val) {
            printf("Maximum value must be greater than or equal to minimum value.\n");
        } else {
            printf("Invalid input. Please enter a number.\n");
        }
        printf("Enter maximum value for elements: ");
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
    printf("Input scalar: ");
    scanf("%lf", &scalar);
    return scalar;
}

void wait_for_enter() {
    printf("Press Enter to continue...\n");
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
                printf("Add two matrices:\n");

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
                printf("Subtract two matrices:\n");

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
                printf("Multiply two matrices:\n");

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
                printf("Multiply matrix by a scalar:\n");

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
                printf("Matrix transposition:\n");

                matrix m = input_matrix_new();
                matrix result = transpose_matrix(m);

                display_matrix(result);

                free_matrix(&m);
                free_matrix(&result);

                wait_for_enter();
                break;
            }

            case 6: {
                printf("Find determinant:\n");

                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("Error: determinant is valid only for square matrices.\n");
                } 
                else {
                    double det = determinant(m);
                    printf("Determinant: %lf\n", det);
                }

                free_matrix(&m);

                wait_for_enter();
                break;
            }
            
            case 7: {
                printf("Find inverse matrix:\n");

                matrix m = input_matrix_new();

                if (m.rows != m.cols) {
                    printf("Error: inverse matrix is valid only for square matrices.\n");
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
                printf("Solve SLE (Ax = B):\n");

                printf("Input matrix A:\n");
                matrix A = input_matrix_new();

                printf("Input vector B:\n");
                matrix B = input_matrix_new();

                if (A.rows != B.rows || B.cols != 1) {
                    printf("Error: B must be a column vector with %d rows.\n", A.rows);
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
                printf("Find rank of a matrix:\n");

                matrix m = input_matrix_new();
                int r = rank(m);

                printf("Rank: %d\n", r);

                free_matrix(&m);

                wait_for_enter();
                break;
            }

            case 10: {
                printf("Generate random matrix:\n");

                int rows, cols;
                double min_val, max_val;

                input_random_matrix_params(&rows, &cols, &min_val, &max_val);
                matrix random_matrix = generate_random_matrix(rows, cols, min_val, max_val);

                printf("Generate random matrix:\n");

                print_matrix(random_matrix);

                free_matrix(&random_matrix);

                wait_for_enter();
                break;
            }

            case 11: {
                printf("Check matrix properites:\n");
                matrix m = input_matrix_new();
                
                printf("Matrix properties:\n");
                printf("Diagonal: %s\n", is_diagonal(m) ? "Yes" : "No");
                printf("Symmetric: %s\n", is_symmetric(m) ? "Yes" : "No");
                printf("Upper triangular: %s\n", is_upper_triangular(m) ? "Yes" : "No");
                printf("Lower triangular: %s\n", is_lower_triangular(m) ? "Yes" : "No");
                printf("Identity: %s\n", is_identity(m) ? "Yes" : "No");

                free_matrix(&m);
                
                wait_for_enter();
                break;
            }

            case 12: {
                printf("Matrix exponentiation:\n");
                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("Error: matrix must be square.\n");
                } else {
                    int exponent;
                    printf("Enter exponent: ");
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
                printf("Cholesky decomposition:\n");
                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("Error: matris must be square.\n");
                } else if (!is_symmetric(m)) {
                    printf("Error: matrix must be symmetric.\n");
                } else {
                    matrix L = cholesky_decomposition(m);
                    printf("Lower triangular matrix L:\n");
                    display_matrix(L);
                    free_matrix(&L);
                }
                
                free_matrix(&m);
                wait_for_enter();
                break;
            }

            case 14: {
                printf("Eigenvalues and Eigenvectors:\n");
                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("Error: matrix must be square.\n");
                } else {
                    double eigenvalue;
                    matrix eigenvector;
                    int max_iter = 1000;
                    double tol = 1e-6;
                    power_method(m, &eigenvalue, &eigenvector, max_iter, tol);
                    printf("Eigenvalue: %f\n", eigenvalue);
                    printf("Eigenvector:\n");
                    display_matrix(eigenvector);
                    free_matrix(&eigenvector);
                }

                free_matrix(&m);
                wait_for_enter();
                break;
            }

            case 15: {
                printf("LU decomposition:\n");
                matrix m = input_matrix_new();
                if (m.rows != m.cols) {
                    printf("Error: matrix must be square");
                } else {
                    matrix L, U;
                    lu_decomposition(m, &L, &U);
                    printf("Lower triangular matrix L:\n");
                    display_matrix(L);
                    printf("Upper triangular matrix U:\n");
                    display_matrix(U);
                    free_matrix(&L);
                    free_matrix(&U);
                }

                free_matrix(&m);
                wait_for_enter();
                break;
            }

            case 16: {
                printf("Matrix norms:\n");
                matrix m = input_matrix_new();
                double f_norm = frobenius_norm(m);
                double one_n = one_norm(m);
                double inf_n = infinity_norm(m);
                printf("Frobenius norm: %f\n", f_norm);
                printf("One-norm: %f\n", one_n);
                printf("Infinity norm: %f\n", inf_n);

                free_matrix(&m);
                wait_for_enter();
                break;
            }
        
            case 17: 
                printf("Exit...\n");
                break;
            default:
                printf("Wrong choice. Please enter a number between 1 and 17.\n");
        }
    } while (choice != 17);
    return 0;
}