#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "operations.h"
#include "ui.h"
#include "utils.h"
#include "matrix_lib/matrix.h"
#include "constants.h"


/**
 * @brief Adds two matrices and displays the result.
 *
 * This function prompts the user to input two matrices, adds them using the `add_matrices` function,
 * and displays the result if the operation is successful. It handles memory allocation and deallocation
 * for the matrices involved.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful, otherwise an error code.
 */
int add_matrices_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m1 = input_matrix_new();
    matrix m2 = input_matrix_new();
    matrix result;
    int error = add_matrices(m1, m2, &result);
    if (error) {
        printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
        free_matrix(&m1);
        free_matrix(&m2);
        return error;
    }
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m1);
    free_matrix(&m2);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Subtracts one matrix from another and displays the result.
 *
 * This function prompts the user to input two matrices, subtracts the second from the first using
 * the `subtract_matrices` function, and displays the result if the operation is successful. It handles
 * memory allocation and deallocation for the matrices involved.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful, otherwise an error code.
 */
int subtract_matrices_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m1 = input_matrix_new();
    matrix m2 = input_matrix_new();
    matrix result;
    int error = subtract_matrices(m1, m2, &result);
    if (error) {
        printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
        free_matrix(&m1);
        free_matrix(&m2);
        return error;
    }
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m1);
    free_matrix(&m2);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Multiplies two matrices using Strassen's algorithm and displays the result.
 *
 * This function prompts the user to input two matrices, multiplies them using the padded version of
 * Strassen's algorithm (`multiply_matrices_strassen_padded`), and displays the result if the operation
 * is successful. It handles memory allocation and deallocation for the matrices involved.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful, otherwise an error code.
 */
int multiply_matrices_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m1 = input_matrix_new();
    matrix m2 = input_matrix_new();
    matrix result;
    int error = multiply_matrices_strassen_padded(m1, m2, &result);
    if (error) {
        printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
        free_matrix(&m1);
        free_matrix(&m2);
        return error;
    }
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m1);
    free_matrix(&m2);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Multiplies a matrix by a scalar and displays the result.
 *
 * This function prompts the user to input a matrix and a scalar, multiplies the matrix by the scalar
 * using the `scalar_multiply` function, and displays the result. It handles memory allocation and
 * deallocation for the matrices involved.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful.
 */
int scalar_multiply_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    double scalar = input_scalar();
    matrix result = scalar_multiply(m, scalar);
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Computes the transpose of a matrix and displays the result.
 *
 * This function prompts the user to input a matrix, computes its transpose using the `transpose_matrix`
 * function, and displays the result. It handles memory allocation and deallocation for the matrices involved.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful.
 */
int transpose_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    matrix result = transpose_matrix(m);
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Computes the determinant of a matrix and displays the result.
 *
 * This function prompts the user to input a matrix, computes its determinant using the `determinant`
 * function, and displays the result if the operation is successful. It handles memory allocation and
 * deallocation for the matrix.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful, otherwise an error code.
 */
int determinant_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    double det;
    int error = determinant(m, &det);
    if (error) {
        printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
        free_matrix(&m);
        return error;
    }
    printf("\n%sDeterminant: %lf\n%s", UGRN, det, COLOR_RESET);
    free_matrix(&m);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Computes the inverse of a matrix and displays the result.
 *
 * This function prompts the user to input a matrix, computes its inverse using the `inverse_matrix`
 * function, and displays the result if the operation is successful. It handles memory allocation and
 * deallocation for the matrices involved.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful, otherwise an error code.
 */
int inverse_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    matrix result;
    int error = inverse_matrix(m, &result);
    if (error) {
        printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
        free_matrix(&m);
        return error;
    }
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Solves a system of linear equations and displays the solution.
 *
 * This function prompts the user to input a coefficient matrix A and a vector B, then solves the system
 * A * x = B using the `solve_system` function. It also computes and displays the condition number of A
 * and chooses the appropriate method (LU or SVD) based on the matrix properties. The solution vector x
 * is displayed if the operation is successful.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful, otherwise an error code.
 */
int solve_system_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix A = input_matrix_new();
    printf("%sInput vector B:\n%s", UBLU, COLOR_RESET);
    matrix B = input_matrix_new();
    if (A.rows != B.rows || B.cols != 1) {
        printf("%sError: B must be a column vector with %d rows.\n%s", URED, A.rows, COLOR_RESET);
        free_matrix(&A);
        free_matrix(&B);
        return INVALID_DIMENSIONS;
    }

    if (A.rows == A.cols) {
        printf("%sSelect norm for condition number:\n1. One-norm\n2. Infinity-norm\n3. Frobenius norm\nEnter choice: %s", UBLU, COLOR_RESET);
        int norm_choice;
        scanf("%d", &norm_choice);
        double (*norm_func)(matrix) = one_norm;
        switch (norm_choice) {
            case 2: norm_func = infinity_norm; break;
            case 3: norm_func = frobenius_norm; break;
            default: norm_func = one_norm;
        }
        double cond = condition_number(A, norm_func);
        if (cond > 0) {
            printf("%sCondition number: %.2e\n%s", UGRN, cond, COLOR_RESET);
            if (cond > 1e6) {
                printf("%sWarning: Matrix is ill-conditioned (condition number > 1e6). Using SVD-based method.\n%s", URED, COLOR_RESET);
            }
        } else {
            printf("%sWarning: Could not compute condition number (matrix may be singular).\n%s", URED, COLOR_RESET);
        }
    } else {
        printf("%sMatrix is not square. Using SVD-based method for solving.\n%s", UGRN, COLOR_RESET);
    }

    matrix x;
    int error = solve_system(A, B, &x);
    if (error) {
        printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
        free_matrix(&A);
        free_matrix(&B);
        return error;
    }
    display_matrix(x);
    free_matrix(&x);
    free_matrix(&A);
    free_matrix(&B);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Computes the rank of a matrix and displays the result.
 *
 * This function prompts the user to input a matrix, computes its rank using the `rank` function,
 * and displays the result. It handles memory allocation and deallocation for the matrix.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful.
 */
int rank_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    int r = rank(m);
    printf("\n%sRank: %d\n%s", UGRN, r, COLOR_RESET);
    free_matrix(&m);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Generates a random matrix based on user input and saves it.
 *
 * This function prompts the user for the dimensions and value range of a random matrix, generates
 * the matrix using `generate_random_matrix`, saves it in `saved_matrix`, and displays it. If a matrix
 * was previously loaded, it is freed before generating the new one.
 *
 * @param saved_matrix Pointer to the matrix where the generated matrix will be stored.
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (set to 1 after generation).
 * @return int Status code: SUCCESS if the operation is successful.
 */
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
    return SUCCESS;
}

/**
 * @brief Checks and displays properties of a matrix.
 *
 * This function prompts the user to input a matrix and checks if it is diagonal, symmetric, upper triangular,
 * lower triangular, or an identity matrix, displaying the results.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful.
 */
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
    return SUCCESS;
}

/**
 * @brief Computes the power of a matrix and displays the result.
 *
 * This function prompts the user to input a matrix and an integer exponent, computes the matrix raised
 * to that power using `matrix_power`, and displays the result if successful.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful, otherwise an error code.
 */
int matrix_power_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    int exponent;
    printf("%sEnter exponent: %s", UBLU, COLOR_RESET);
    scanf("%d", &exponent);
    while (getchar() != '\n');
    matrix result;
    int error = matrix_power(m, exponent, &result);
    if (error) {
        printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
        free_matrix(&m);
        return error;
    }
    display_matrix(result);
    free_matrix(&result);
    free_matrix(&m);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Performs Cholesky decomposition on a matrix and displays the lower triangular matrix.
 *
 * This function prompts the user to input a symmetric positive-definite matrix, performs Cholesky
 * decomposition using `cholesky_decomposition`, and displays the resulting lower triangular matrix L.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful, otherwise an error code.
 */
int cholesky_decomposition_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    matrix L;
    int error = cholesky_decomposition(m, &L);
    if (error) {
        printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
        free_matrix(&m);
        return error;
    }
    printf("%sLower triangular matrix L:\n%s", UGRN, COLOR_RESET);
    display_matrix(L);
    free_matrix(&L);
    free_matrix(&m);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Computes the eigenvalues and eigenvectors of a matrix and displays them.
 *
 * This function prompts the user to input a matrix, computes its eigenvalues and eigenvectors using
 * the QR algorithm (`qr_algorithm`), and displays the results if successful.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful, otherwise an error code.
 */
int eigenvalues_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    matrix eigenvalues, eigenvectors;
    int max_iter = 2000;
    double tol = 1e-10;
    int error = qr_algorithm(m, &eigenvalues, &eigenvectors, max_iter, tol);
    if (error) {
        printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
        free_matrix(&m);
        return error;
    }
    display_eigen(eigenvalues, eigenvectors);
    free_matrix(&eigenvalues);
    free_matrix(&eigenvectors);
    free_matrix(&m);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Performs LU decomposition on a matrix and displays the L and U matrices.
 *
 * This function prompts the user to input a square matrix, performs LU decomposition using
 * `lu_decomposition`, and displays the lower triangular matrix L and upper triangular matrix U if successful.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful, otherwise an error code.
 */
int lu_decomposition_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    if (m.rows != m.cols) {
        printf("%sError: Matrix must be square.\n%s", URED, COLOR_RESET);
        free_matrix(&m);
        return INVALID_DIMENSIONS;
    }
    matrix L, U;
    int status = lu_decomposition(m, &L, &U);
    if (status) {
        printf("%sError: LU decomposition failed with code %d.\n%s", URED, status, COLOR_RESET);
        free_matrix(&m);
        return status;
    }
    printf("%sLower triangular matrix L:\n%s", UGRN, COLOR_RESET);
    display_matrix(L);
    printf("%sUpper triangular matrix U:\n%s", UGRN, COLOR_RESET);
    display_matrix(U);
    free_matrix(&L);
    free_matrix(&U);
    free_matrix(&m);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Computes and displays various norms of a matrix.
 *
 * This function prompts the user to input a matrix and computes its Frobenius norm, one-norm, and
 * infinity-norm using the respective functions, then displays the results.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful.
 */
int matrix_norms_session_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    double f_norm = frobenius_norm(m);
    double one_n = one_norm(m);
    double inf_n = infinity_norm(m);
    printf("\n%sFrobenius norm: %f\n%s", UGRN, f_norm, COLOR_RESET);
    printf("%sOne-norm: %f\n%s", UGRN, one_n, COLOR_RESET);
    printf("%sInfinity norm: %f\n%s", UGRN, inf_n, COLOR_RESET);
    free_matrix(&m);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Saves a matrix to a file.
 *
 * This function prompts the user to input a matrix and a filename, then saves the matrix to the specified
 * file using `save_matrix_to_file`.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful.
 */
int save_matrix_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    char filename[100];
    printf("%sEnter file name: %s", UBLU, COLOR_RESET);
    scanf("%s", filename);
    while (getchar() != '\n');
    save_matrix_to_file(m, filename);
    free_matrix(&m);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Loads a matrix from a file and saves it for further use.
 *
 * This function prompts the user for a filename, loads the matrix from the file using `load_matrix_from_file`,
 * and stores it in `saved_matrix`. If a matrix was previously loaded, it is freed first.
 *
 * @param saved_matrix Pointer to the matrix where the loaded matrix will be stored.
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (set to 1 after loading).
 * @return int Status code: SUCCESS if the operation is successful, INVALID_INPUT if loading fails.
 */
int load_matrix_operation(matrix *saved_matrix, int *matrix_loaded) {
    char filename[100];
    printf("%sEnter file name: %s", UBLU, COLOR_RESET);
    scanf("%s", filename);
    while (getchar() != '\n');
    if (*matrix_loaded) free_matrix(saved_matrix);
    *saved_matrix = load_matrix_from_file(filename);
    if (saved_matrix->data == NULL) {
        printf("%sError: Failed to load matrix.\n%s", URED, COLOR_RESET);
        *matrix_loaded = 0;
        return INVALID_INPUT;
    }
    *matrix_loaded = 1;
    display_matrix(*saved_matrix);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Performs operations on a loaded matrix.
 *
 * This function allows the user to perform various operations on a previously loaded or generated matrix.
 * It presents a submenu of available operations and executes the selected operation on `saved_matrix`.
 *
 * @param saved_matrix Pointer to the loaded matrix to operate on.
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded.
 * @return int Status code: SUCCESS if the operation is successful, INVALID_INPUT if no matrix is loaded.
 */
int use_loaded_matrix_operation(matrix *saved_matrix, int *matrix_loaded) {
    if (!*matrix_loaded) {
        printf("%sNo matrix loaded or generated. Please load or generate a matrix first.\n%s", URED, COLOR_RESET);
        wait_for_enter();
        return INVALID_INPUT;
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
    while (getchar() != '\n');

    switch (op_choice) {
        case 1: {
            matrix b = input_matrix_new();
            if (saved_matrix->rows != b.rows || saved_matrix->cols != b.cols) {
                printf("%sError: Matrices must have the same dimensions for addition.\n%s", URED, COLOR_RESET);
            } else {
                matrix result;
                int error = add_matrices(*saved_matrix, b, &result);
                if (error == SUCCESS) {
                    display_matrix(result);
                    free_matrix(&result);
                } else {
                    printf("%sAddition failed with error code %d.\n%s", URED, error, COLOR_RESET);
                }
            }
            free_matrix(&b);
            break;
        }
        case 2: {
            matrix b = input_matrix_new();
            if (saved_matrix->rows != b.rows || saved_matrix->cols != b.cols) {
                printf("%sError: Matrices must have the same dimensions for subtraction.\n%s", URED, COLOR_RESET);
            } else {
                matrix result;
                int error = subtract_matrices(*saved_matrix, b, &result);
                if (error == SUCCESS) {
                    display_matrix(result);
                    free_matrix(&result);
                } else {
                    printf("%sSubtraction failed with error code %d.\n%s", URED, error, COLOR_RESET);
                }
            }
            free_matrix(&b);
            break;
        }
        case 3: {
            matrix b = input_matrix_new();
            if (saved_matrix->cols != b.rows) {
                printf("%sError: Number of columns in first matrix must equal number of rows in second matrix.\n%s", URED, COLOR_RESET);
            } else {
                matrix result;
                int error = multiply_matrices_strassen_padded(*saved_matrix, b, &result);
                if (error == SUCCESS) {
                    display_matrix(result);
                    free_matrix(&result);
                } else {
                    printf("%sMultiplication failed with error code %d.\n%s", URED, error, COLOR_RESET);
                }
            }
            free_matrix(&b);
            break;
        }
        case 4: {
            double scalar = input_scalar();
            matrix result = scalar_multiply(*saved_matrix, scalar);
            display_matrix(result);
            free_matrix(&result);
            break;
        }
        case 5: {
            matrix result = transpose_matrix(*saved_matrix);
            display_matrix(result);
            free_matrix(&result);
            break;
        }
        case 6: {
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Determinant is only defined for square matrices.\n%s", URED, COLOR_RESET);
            } else {
                double det;
                int error = determinant(*saved_matrix, &det);
                if (error == SUCCESS) {
                    printf("\n%sDeterminant: %lf\n%s", UGRN, det, COLOR_RESET);
                } else {
                    printf("%sDeterminant calculation failed with error code %d.\n%s", URED, error, COLOR_RESET);
                }
            }
            break;
        }
        case 7: {
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Inverse is only defined for square matrices.\n%s", URED, COLOR_RESET);
            } else {
                matrix result;
                int error = inverse_matrix(*saved_matrix, &result);
                if (error == SUCCESS) {
                    display_matrix(result);
                    free_matrix(&result);
                } else {
                    printf("%sInverse calculation failed with error code %d.\n%s", URED, error, COLOR_RESET);
                }
            }
            break;
        }
        case 8: {
            printf("%sInput vector B:\n%s", UBLU, COLOR_RESET);
            matrix b = input_matrix_new();
            if (saved_matrix->rows != b.rows || b.cols != 1) {
                printf("%sError: B must be a column vector with %d rows.\n%s", URED, saved_matrix->rows, COLOR_RESET);
            } else {
                matrix x;
                int error = solve_system(*saved_matrix, b, &x);
                if (error == SUCCESS) {
                    display_matrix(x);
                    free_matrix(&x);
                } else {
                    printf("%sSolving system failed with error code %d.\n%s", URED, error, COLOR_RESET);
                }
            }
            free_matrix(&b);
            break;
        }
        case 9: {
            int r = rank(*saved_matrix);
            printf("\n%sRank: %d\n%s", UGRN, r, COLOR_RESET);
            break;
        }
        case 11: {
            printf("\n%sMatrix properties:\n%s", UGRN, COLOR_RESET);
            printf("%sDiagonal: %s\n%s", UGRN, is_diagonal(*saved_matrix) ? "Yes" : "No", COLOR_RESET);
            printf("%sSymmetric: %s\n%s", UGRN, is_symmetric(*saved_matrix) ? "Yes" : "No", COLOR_RESET);
            printf("%sUpper triangular: %s\n%s", UGRN, is_upper_triangular(*saved_matrix) ? "Yes" : "No", COLOR_RESET);
            printf("%sLower triangular: %s\n%s", UGRN, is_lower_triangular(*saved_matrix) ? "Yes" : "No", COLOR_RESET);
            printf("%sIdentity: %s\n%s", UGRN, is_identity(*saved_matrix) ? "Yes" : "No", COLOR_RESET);
            break;
        }
        case 12: {
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Matrix must be square for exponentiation.\n%s", URED, COLOR_RESET);
            } else {
                int exponent;
                printf("%sEnter exponent: %s", UBLU, COLOR_RESET);
                scanf("%d", &exponent);
                while (getchar() != '\n');
                matrix result;
                int error = matrix_power(*saved_matrix, exponent, &result);
                if (error == SUCCESS) {
                    display_matrix(result);
                    free_matrix(&result);
                } else {
                    printf("%sMatrix power failed with error code %d.\n%s", URED, error, COLOR_RESET);
                }
            }
            break;
        }
        case 13: {
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Matrix must be square for Cholesky decomposition.\n%s", URED, COLOR_RESET);
            } else if (!is_symmetric(*saved_matrix)) {
                printf("%sError: Matrix must be symmetric.\n%s", URED, COLOR_RESET);
            } else {
                matrix L;
                int error = cholesky_decomposition(*saved_matrix, &L);
                if (error == SUCCESS) {
                    printf("%sLower triangular matrix L:\n%s", UGRN, COLOR_RESET);
                    display_matrix(L);
                    free_matrix(&L);
                } else {
                    printf("%sCholesky decomposition failed with error code %d.\n%s", URED, error, COLOR_RESET);
                }
            }
            break;
        }
        case 14: {
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Matrix must be square.\n%s", URED, COLOR_RESET);
            } else {
                matrix eigenvalues, eigenvectors;
                int max_iter = 2000;
                double tol = 1e-10;
                int error = qr_algorithm(*saved_matrix, &eigenvalues, &eigenvectors, max_iter, tol);
                if (error == SUCCESS) {
                    display_eigen(eigenvalues, eigenvectors);
                    free_matrix(&eigenvalues);
                    free_matrix(&eigenvectors);
                } else {
                    printf("%sEigenvalue calculation failed with error code %d.\n%s", URED, error, COLOR_RESET);
                }
            }
            break;
        }
        case 15: {
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Matrix must be square.\n%s", URED, COLOR_RESET);
            } else {
                matrix L, U;
                int status = lu_decomposition(*saved_matrix, &L, &U);
                if (status == SUCCESS) {
                    printf("%sLower triangular matrix L:\n%s", UGRN, COLOR_RESET);
                    display_matrix(L);
                    printf("%sUpper triangular matrix U:\n%s", UGRN, COLOR_RESET);
                    display_matrix(U);
                    free_matrix(&L);
                    free_matrix(&U);
                } else {
                    printf("%sLU decomposition failed with error code %d.\n%s", URED, status, COLOR_RESET);
                }
            }
            break;
        }
        case 16: {
            double f_norm = frobenius_norm(*saved_matrix);
            double one_n = one_norm(*saved_matrix);
            double inf_n = infinity_norm(*saved_matrix);
            printf("\n%sFrobenius norm: %f\n%s", UGRN, f_norm, COLOR_RESET);
            printf("%sOne-norm: %f\n%s", UGRN, one_n, COLOR_RESET);
            printf("%sInfinity norm: %f\n%s", UGRN, inf_n, COLOR_RESET);
            break;
        }
        case 21: {
            matrix U, Sigma, V;
            int error = svd(*saved_matrix, &U, &Sigma, &V);
            if (error == SUCCESS) {
                printf("%sMatrix U:%s\n", UGRN, COLOR_RESET);
                display_matrix(U);
                printf("%sMatrix Sigma:%s\n", UGRN, COLOR_RESET);
                display_matrix(Sigma);
                printf("%sMatrix V:%s\n", UGRN, COLOR_RESET);
                display_matrix(V);
                free_matrix(&U);
                free_matrix(&Sigma);
                free_matrix(&V);
            } else {
                printf("%sSVD failed with error code %d.\n%s", URED, error, COLOR_RESET);
            }
            break;
        }
        case 22: {
            if (saved_matrix->rows != saved_matrix->cols) {
                printf("%sError: Matrix must be square.\n%s", URED, COLOR_RESET);
            } else {
                matrix Q, T;
                int max_iter = 1000;
                double tol = 1e-8;
                int error = schur_decomposition(*saved_matrix, &Q, &T, max_iter, tol);
                if (error == SUCCESS) {
                    printf("%sOrthogonal matrix Q:\n%s", UGRN, COLOR_RESET);
                    display_matrix(Q);
                    printf("%sQuasi-triangular matrix T:\n%s", UGRN, COLOR_RESET);
                    display_matrix(T);
                    free_matrix(&Q);
                    free_matrix(&T);
                } else {
                    printf("%sSchur decomposition failed with error code %d.\n%s", URED, error, COLOR_RESET);
                }
            }
            break;
        }
        default:
            printf("%sInvalid operation choice.\n%s", URED, COLOR_RESET);
    }
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Saves the currently loaded random matrix to a file.
 *
 * This function prompts the user for a filename and saves the currently loaded random matrix to that file.
 * If no random matrix is loaded, it displays an error message.
 *
 * @param saved_matrix Pointer to the matrix to be saved.
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded.
 * @return int Status code: SUCCESS if the operation is successful, INVALID_INPUT if no matrix is loaded.
 */
int save_random_matrix_operation(matrix *saved_matrix, int *matrix_loaded) {
    if (!*matrix_loaded) {
        printf("%sNo random matrix generated. Please generate a matrix first.\n%s", URED, COLOR_RESET);
        wait_for_enter();
        return INVALID_INPUT;
    }
    char filename[100];
    printf("%sEnter file name: %s", UBLU, COLOR_RESET);
    scanf("%s", filename);
    while (getchar() != '\n');
    save_matrix_to_file(*saved_matrix, filename);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Performs Singular Value Decomposition (SVD) on a matrix and displays the results.
 *
 * This function prompts the user to input a matrix, performs SVD using `svd`, and displays the matrices
 * U, Sigma, and V if successful.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful, otherwise an error code.
 */
int svd_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    matrix U, Sigma, V;
    int error = svd(m, &U, &Sigma, &V);
    if (error == SUCCESS) {
        printf("\n%sSingular value decomposition (SVD):%s\n", UGRN, COLOR_RESET);
        printf("%sMatrix U:%s\n", UGRN, COLOR_RESET);
        display_matrix(U);
        printf("%sMatrix Sigma:%s\n", UGRN, COLOR_RESET);
        display_matrix(Sigma);
        printf("%sMatrix V:%s\n", UGRN, COLOR_RESET);
        display_matrix(V);
        free_matrix(&U);
        free_matrix(&Sigma);
        free_matrix(&V);
    } else {
        printf("%sSVD failed with error code %d.\n%s", URED, error, COLOR_RESET);
    }
    free_matrix(&m);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Performs Schur decomposition on a matrix and displays the results.
 *
 * This function prompts the user to input a square matrix, performs Schur decomposition using
 * `schur_decomposition`, and displays the orthogonal matrix Q and quasi-triangular matrix T if successful.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS if the operation is successful, otherwise an error code.
 */
int schur_decomposition_operation(matrix *saved_matrix, int *matrix_loaded) {
    matrix m = input_matrix_new();
    if (m.rows != m.cols) {
        printf("%sError: Matrix must be square.\n%s", URED, COLOR_RESET);
        free_matrix(&m);
        return INVALID_DIMENSIONS;
    }
    matrix Q, T;
    int max_iter = 1000;
    double tol = 1e-8;
    int error = schur_decomposition(m, &Q, &T, max_iter, tol);
    if (error == SUCCESS) {
        printf("%sOrthogonal matrix Q:\n%s", UGRN, COLOR_RESET);
        display_matrix(Q);
        printf("%sQuasi-triangular matrix T:\n%s", UGRN, COLOR_RESET);
        display_matrix(T);
        free_matrix(&Q);
        free_matrix(&T);
    } else {
        printf("%sSchur decomposition failed with error code %d.\n%s", URED, error, COLOR_RESET);
    }
    free_matrix(&m);
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Reduces a matrix to Hessenberg form and displays the results.
 *
 * This function either prompts the user to input a matrix or uses the loaded matrix, reduces it to
 * Hessenberg form using `hessenberg_form`, and displays the Hessenberg matrix H and orthogonal matrix Q.
 *
 * @param saved_matrix Pointer to the loaded matrix (used if `matrix_loaded` is true).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded.
 * @return int Status code: SUCCESS if the operation is successful, otherwise an error code.
 */
int hessenberg_form_operation(matrix *saved_matrix, int *matrix_loaded) {
    if (!*matrix_loaded) {
        matrix m = input_matrix_new();
        matrix H, Q;
        int error = hessenberg_form(m, &H, &Q);
        if (error) {
            printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
            free_matrix(&m);
            return error;
        }
        printf("%sHessenberg matrix H:\n%s", UGRN, COLOR_RESET);
        display_matrix(H);
        printf("%sOrthogonal matrix Q:\n%s", UGRN, COLOR_RESET);
        display_matrix(Q);
        free_matrix(&H);
        free_matrix(&Q);
        free_matrix(&m);
    } else {
        matrix H, Q;
        int error = hessenberg_form(*saved_matrix, &H, &Q);
        if (error) {
            printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
            return error;
        }
        printf("%sHessenberg matrix H:\n%s", UGRN, COLOR_RESET);
        display_matrix(H);
        printf("%sOrthogonal matrix Q:\n%s", UGRN, COLOR_RESET);
        display_matrix(Q);
        free_matrix(&H);
        free_matrix(&Q);
    }
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Generates a special type of matrix based on user input.
 *
 * This function prompts the user to select a type of special matrix (Hilbert, Vandermonde, Toeplitz,
 * Hadamard, or Jacobi), inputs the necessary parameters, generates the matrix, and saves it in `saved_matrix`.
 *
 * @param saved_matrix Pointer to the matrix where the generated matrix will be stored.
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (set to 1 after generation).
 * @return int Status code: SUCCESS if the operation is successful, INVALID_INPUT if the type is invalid.
 */
int generate_special_matrix_operation(matrix *saved_matrix, int *matrix_loaded) {
    int type;
    printf("%sTypes of special matrices:%s\n\n", UCYN, COLOR_RESET);
    printf("%s1. Hilbert matrix%s\n", UYEL, COLOR_RESET);
    printf("%s2. Vandermonde matrix%s\n", UYEL, COLOR_RESET);
    printf("%s3. Toeplitz matrix%s\n", UYEL, COLOR_RESET);
    printf("%s4. Hadamard matrix%s\n", UYEL, COLOR_RESET);
    printf("%s5. Jacobi matrix%s\n", UYEL, COLOR_RESET);
    printf("%sEnter type: %s", UBLU, COLOR_RESET);
    scanf("%d", &type);
    while (getchar() != '\n');

    int n;
    switch (type) {
        case 1: { 
            n = input_positive_integer("Input size of Hilbert's matrix: ");
            if (*matrix_loaded) free_matrix(saved_matrix);
            int error = generate_hilbert_matrix(n, saved_matrix);
            if (error == SUCCESS) {
                *matrix_loaded = 1;
                printf("%sHilbert's matrix:\n%s", UGRN, COLOR_RESET);
                display_matrix(*saved_matrix);
            } else {
                printf("%sFailed to generate Hilbert matrix.\n%s", URED, COLOR_RESET);
            }
            break;
        }
        case 2: { 
            n = input_positive_integer("Input size of Vandermonde's matrix: ");
            double* values = malloc(n * sizeof(double));
            printf("%sInput %d values for Vandermonde's matrix:\n%s", UBLU, n, COLOR_RESET);
            for (int i = 0; i < n; i++) {
                printf("Value %d: ", i + 1);
                scanf("%lf", &values[i]);
            }
            while (getchar() != '\n');
            if (*matrix_loaded) free_matrix(saved_matrix);
            int error = generate_vandermonde_matrix(n, values, saved_matrix);
            if (error == SUCCESS) {
                *matrix_loaded = 1;
                printf("%sVandermonde's matrix:\n%s", UGRN, COLOR_RESET);
                display_matrix(*saved_matrix);
            } else {
                printf("%sFailed to generate Vandermonde matrix.\n%s", URED, COLOR_RESET);
            }
            free(values);
            break;
        }
        case 3: { 
            n = input_positive_integer("Input size of Toeplitz's matrix: ");
            double* row = malloc(n * sizeof(double));
            printf("%sInput %d values for first row of matrix:\n%s", UBLU, n, COLOR_RESET);
            for (int i = 0; i < n; i++) {
                printf("Value %d: ", i + 1);
                scanf("%lf", &row[i]);
            }
            while (getchar() != '\n');
            if (*matrix_loaded) free_matrix(saved_matrix);
            int error = generate_toeplitz_matrix(n, row, saved_matrix);
            if (error == SUCCESS) {
                *matrix_loaded = 1;
                printf("%sToeplitz's matrix:\n%s", UGRN, COLOR_RESET);
                display_matrix(*saved_matrix);
            } else {
                printf("%sFailed to generate Toeplitz matrix.\n%s", URED, COLOR_RESET);
            }
            free(row);
            break;
        }
        case 4: { 
            n = input_positive_integer("Input size of Hadamard's matrix (must be power of 2): ");
            if (*matrix_loaded) free_matrix(saved_matrix);
            int error = generate_hadamard_matrix(n, saved_matrix);
            if (error) {
                printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
                return error;
            }
            *matrix_loaded = 1;
            printf("%sHadamard's matrix:\n%s", UGRN, COLOR_RESET);
            display_matrix(*saved_matrix);
            break;
        }
        case 5: { 
            n = input_positive_integer("Input size of Jacobi's matrix: ");
            double a, b;
            printf("%sInput value for main diagonal (a): %s", UBLU, COLOR_RESET);
            scanf("%lf", &a);
            printf("%sInput value for adjacent diagonals (b): %s", UBLU, COLOR_RESET);
            scanf("%lf", &b);
            while (getchar() != '\n');
            if (*matrix_loaded) free_matrix(saved_matrix);
            int error = generate_jacobi_matrix(n, a, b, saved_matrix);
            if (error == SUCCESS) {
                *matrix_loaded = 1;
                printf("%sJacobi's matrix:\n%s", UGRN, COLOR_RESET);
                display_matrix(*saved_matrix);
            } else {
                printf("%sFailed to generate Jacobi matrix.\n%s", URED, COLOR_RESET);
            }
            break;
        }
        default:
            printf("%sIncorrect input.\n%s", URED, COLOR_RESET);
            return INVALID_INPUT;
    }
    wait_for_enter();
    return SUCCESS;
}

/**
 * @brief Exits the program.
 *
 * This function prints an exit message and returns SUCCESS to indicate the program should terminate.
 *
 * @param saved_matrix Pointer to a matrix that can be used across operations (not used in this function).
 * @param matrix_loaded Pointer to an integer indicating if a matrix is loaded (not used in this function).
 * @return int Status code: SUCCESS to indicate successful exit.
 */
int exit_operation(matrix *saved_matrix, int *matrix_loaded) {
    printf("\nExit...\n");
    return SUCCESS;
}