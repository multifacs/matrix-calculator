#include <check.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

// Addition test
START_TEST(test_add_matrices) {

    matrix a = create_matrix(2, 2);
    a.data[0][0] = 1; a.data[0][1] = 2;
    a.data[1][0] = 3; a.data[1][1] = 4;

    matrix b = create_matrix(2, 2);
    b.data[0][0] = 5; b.data[0][1] = 6;
    b.data[1][0] = 7; b.data[1][1] = 8;

    matrix expected = create_matrix(2, 2);
    expected.data[0][0] = 6; expected.data[0][1] = 8;
    expected.data[1][0] = 10; expected.data[1][1] = 12;

    matrix result = add_matrices(a, b);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Subtraction test
START_TEST(test_subtract_matrices) {

    matrix a = create_matrix(2, 2);
    a.data[0][0] = 5; a.data[0][1] = 6;
    a.data[1][0] = 7; a.data[1][1] = 8;

    matrix b = create_matrix(2, 2);
    b.data[0][0] = 1; b.data[0][1] = 2;
    b.data[1][0] = 3; b.data[1][1] = 4;

    matrix expected = create_matrix(2, 2);
    expected.data[0][0] = 4; expected.data[0][1] = 4;
    expected.data[1][0] = 4; expected.data[1][1] = 4;

    matrix result = subtract_matrices(a, b);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Multiplication test (square matrices with the size of two (Strassen))
START_TEST(test_multiply_matrices) {

    matrix a = create_matrix(2, 2);
    a.data[0][0] = 1; a.data[0][1] = 2;
    a.data[1][0] = 3; a.data[1][1] = 4;

    matrix b = create_matrix(2, 2);
    b.data[0][0] = 5; b.data[0][1] = 6;
    b.data[1][0] = 7; b.data[1][1] = 8;

    matrix expected = create_matrix(2, 2);
    expected.data[0][0] = 19; expected.data[0][1] = 22;
    expected.data[1][0] = 43; expected.data[1][1] = 50;

    matrix result = multiply_matrices_strassen_padded(a, b);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Multiplication test (non-square matrices (standard multiplication))
START_TEST(test_multiply_matrices_non_square) {

    matrix a = create_matrix(2, 3);
    a.data[0][0] = 1; a.data[0][1] = 2; a.data[0][2] = 3;
    a.data[1][0] = 4; a.data[1][1] = 5; a.data[1][2] = 6;

    matrix b = create_matrix(3, 2);
    b.data[0][0] = 7; b.data[0][1] = 8;
    b.data[1][0] = 9; b.data[1][1] = 10;
    b.data[2][0] = 11; b.data[2][1] = 12;

    matrix expected = create_matrix(2, 2);
    expected.data[0][0] = 58; expected.data[0][1] = 64;
    expected.data[1][0] = 139; expected.data[1][1] = 154;

    matrix result = multiply_matrices_strassen_padded(a, b);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Multiplication test for square matrices with size not power of two (standard multiplication)
START_TEST(test_multiply_matrices_square_not_power_of_two) {

    matrix a = create_matrix(3, 3);
    a.data[0][0] = 1; a.data[0][1] = 2; a.data[0][2] = 3;
    a.data[1][0] = 4; a.data[1][1] = 5; a.data[1][2] = 6;
    a.data[2][0] = 7; a.data[2][1] = 8; a.data[2][2] = 9;

    matrix b = create_matrix(3, 3);
    b.data[0][0] = 9; b.data[0][1] = 8; b.data[0][2] = 7;
    b.data[1][0] = 6; b.data[1][1] = 5; b.data[1][2] = 4;
    b.data[2][0] = 3; b.data[2][1] = 2; b.data[2][2] = 1;

    matrix expected = create_matrix(3, 3);
    expected.data[0][0] = 30; expected.data[0][1] = 24; expected.data[0][2] = 18;
    expected.data[1][0] = 84; expected.data[1][1] = 69; expected.data[1][2] = 54;
    expected.data[2][0] = 138; expected.data[2][1] = 114; expected.data[2][2] = 90;

    matrix result = multiply_matrices_strassen_padded(a, b);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Multiplication test for large square matrices with size power of two (Strassen)
START_TEST(test_multiply_matrices_large) {
    matrix a = create_matrix(64, 64);
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            a.data[i][j] = i * 64 + j + 1;
        }
    }

    matrix b = create_matrix(64, 64);
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            b.data[i][j] = (i * 64 + j + 1) * 2;
        }
    }

    matrix expected = create_matrix(64, 64);
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            expected.data[i][j] = 0;
            for (int k = 0; k < 64; k++) {
                expected.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }

    matrix result = multiply_matrices_strassen_padded(a, b);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Test for QR algorithm accuracy
START_TEST(test_qr_algorithm) {
    matrix a = create_matrix(2, 2);
    a.data[0][0] = 2; a.data[0][1] = 1;
    a.data[1][0] = 1; a.data[1][1] = 2;

    matrix eigenvalues, eigenvectors;
    qr_algorithm(a, &eigenvalues, &eigenvectors, 1000, 1e-6);

    // Expected eigenvalues: 3 and 1 (for symmetric matrix [[2,1],[1,2]])
    ck_assert_double_eq_tol(eigenvalues.data[0][0], 3.0, 1e-2);
    ck_assert_double_eq_tol(eigenvalues.data[1][0], 1.0, 1e-2);

    free_matrix(&a);
    free_matrix(&eigenvalues);
    free_matrix(&eigenvectors);
}
END_TEST

// Scalar multiplication test
START_TEST(test_scalar_multiply) {

    matrix a = create_matrix(2, 2);
    a.data[0][0] = 1; a.data[0][1] = 2;
    a.data[1][0] = 3; a.data[1][1] = 4;

    double scalar = 2.5;

    matrix expected = create_matrix(2, 2);
    expected.data[0][0] = 2.5; expected.data[0][1] = 5.0;
    expected.data[1][0] = 7.5; expected.data[1][1] = 10.0;

    matrix result = scalar_multiply(a, scalar);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&a);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Transpose test
START_TEST(test_transpose_matrix) {

    matrix a = create_matrix(2, 3);
    a.data[0][0] = 1; a.data[0][1] = 2; a.data[0][2] = 3;
    a.data[1][0] = 4; a.data[1][1] = 5; a.data[1][2] = 6;

    matrix expected = create_matrix(3, 2);
    expected.data[0][0] = 1; expected.data[0][1] = 4;
    expected.data[1][0] = 2; expected.data[1][1] = 5;
    expected.data[2][0] = 3; expected.data[2][1] = 6;

    matrix result = transpose_matrix(a);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&a);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Determinant evaluation test
START_TEST(test_determinant) {

    matrix a = create_matrix(2, 2);
    a.data[0][0] = 1; a.data[0][1] = 2;
    a.data[1][0] = 3; a.data[1][1] = 4;

    double det = determinant(a);

    ck_assert_double_eq_tol(det, -2.0, 1e-6);

    free_matrix(&a);
}
END_TEST

// Inverse matrix test
START_TEST(test_inverse_matrix) {

    matrix a = create_matrix(2, 2);
    a.data[0][0] = 4; a.data[0][1] = 7;
    a.data[1][0] = 2; a.data[1][1] = 6;

    matrix expected = create_matrix(2, 2);
    expected.data[0][0] = 0.6; expected.data[0][1] = -0.7;
    expected.data[1][0] = -0.2; expected.data[1][1] = 0.4;

    matrix result = inverse_matrix(a);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&a);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Find solution for SLE test
START_TEST(test_solve_system) {

    matrix A = create_matrix(2, 2);
    A.data[0][0] = 7; A.data[0][1] = 2;
    A.data[1][0] = 17; A.data[1][1] = 6;

    matrix B = create_matrix(2, 1);
    B.data[0][0] = 1;
    B.data[1][0] = -9;

    matrix expected = create_matrix(2, 1);
    expected.data[0][0] = 3.0;
    expected.data[1][0] = -10.0;

    matrix result = solve_system(A, B);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Evaluate rank of a matrix
START_TEST(test_rank) {

    matrix a = create_matrix(3, 3);
    a.data[0][0] = 1; a.data[0][1] = 2; a.data[0][2] = 3;
    a.data[1][0] = 4; a.data[1][1] = 5; a.data[1][2] = 6;
    a.data[2][0] = 7; a.data[2][1] = 8; a.data[2][2] = 9;

    int r = rank(a);

    ck_assert_int_eq(r, 2);
    
    free_matrix(&a);
}
END_TEST

// Test for random matrix generation
START_TEST(test_generate_random_matrix) {

    int rows = 2, cols = 2;
    double min_val = 0.0, max_val = 10.0;

    matrix random_matrix = generate_random_matrix(rows, cols, min_val, max_val);

    ck_assert_int_eq(random_matrix.rows, rows);
    ck_assert_int_eq(random_matrix.cols, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ck_assert(random_matrix.data[i][j] >= min_val && random_matrix.data[i][j] <= max_val);
        }
    }

    free_matrix(&random_matrix);
}
END_TEST

// Test for matrix exponentiation (positive power)
START_TEST(test_matrix_power_positive) {

    matrix a = create_matrix(2, 2);
    a.data[0][0] = 3; a.data[0][1] = 5;
    a.data[1][0] = 7; a.data[1][1] = 2;

    matrix expected = create_matrix(2, 2);
    expected.data[0][0] = 44; expected.data[0][1] = 25;
    expected.data[1][0] = 35; expected.data[1][1] = 39;

    matrix result = matrix_power(a, 2);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&a);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Test for matrix exponentiation (negative)
START_TEST(test_matrix_power_negative) {

    matrix a = create_matrix(2, 2);
    a.data[0][0] = 7; a.data[0][1] = 5;
    a.data[1][0] = 4; a.data[1][1] = 3;

    matrix expected = create_matrix(2, 2);
    expected.data[0][0] = 29; expected.data[0][1] = -50;
    expected.data[1][0] = -40; expected.data[1][1] = 69;

    matrix result = matrix_power(a, -2);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&a);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Test for cholesky decomposition
START_TEST(test_cholesky_decomposition) {
    matrix a = create_matrix(2, 2);
    a.data[0][0] = 4; a.data[0][1] = 2;
    a.data[1][0] = 2; a.data[1][1] = 5;

    matrix L = cholesky_decomposition(a);
    matrix Lt = transpose_matrix(L);
    matrix product = multiply_matrices(L, Lt);

    ck_assert_int_eq(matrices_equal(product, a), 1);

    free_matrix(&a);
    free_matrix(&L);
    free_matrix(&Lt);
    free_matrix(&product);
}
END_TEST

// Test for LU decomposition
START_TEST(test_lu_decomposition) {
    matrix a = create_matrix(2, 2);
    a.data[0][0] = 4; a.data[0][1] = 3;
    a.data[1][0] = 6; a.data[1][1] = 3;

    matrix L, U;
    lu_decomposition(a, &L, &U);
    matrix product = multiply_matrices(L, U);

    ck_assert_int_eq(matrices_equal(product, a), 1);

    free_matrix(&a);
    free_matrix(&L);
    free_matrix(&U);
    free_matrix(&product);
}
END_TEST

// Test for matrix norms
START_TEST(test_matrix_norms) {
    matrix a = create_matrix(2, 2);
    a.data[0][0] = 1; a.data[0][1] = 2;
    a.data[1][0] = 3; a.data[1][1] = 4;

    double f_norm = frobenius_norm(a);
    double one_n = one_norm(a);
    double inf_n = infinity_norm(a);

    ck_assert_double_eq_tol(f_norm, sqrt(30), 1e-6);
    ck_assert_double_eq_tol(one_n, 6, 1e-6);
    ck_assert_double_eq_tol(inf_n, 7, 1e-6);

    free_matrix(&a);
}
END_TEST

// Test for is_diagonal
START_TEST(test_is_diagonal) {
    matrix diag = create_matrix(2, 2);
    diag.data[0][0] = 1; diag.data[0][1] = 0;
    diag.data[1][0] = 0; diag.data[1][1] = 1;
    ck_assert_int_eq(is_diagonal(diag), 1);

    matrix not_diag = create_matrix(2, 2);
    not_diag.data[0][0] = 1; not_diag.data[0][1] = 2;
    not_diag.data[1][0] = 3; not_diag.data[1][1] = 4;
    ck_assert_int_eq(is_diagonal(not_diag), 0);

    free_matrix(&diag);
    free_matrix(&not_diag);
}
END_TEST

// Test for is_symmetric
START_TEST(test_is_symmetric) {
    matrix sym = create_matrix(2, 2);
    sym.data[0][0] = 1; sym.data[0][1] = 2;
    sym.data[1][0] = 2; sym.data[1][1] = 3;
    ck_assert_int_eq(is_symmetric(sym), 1);

    matrix not_sym = create_matrix(2, 2);
    not_sym.data[0][0] = 1; not_sym.data[0][1] = 2;
    not_sym.data[1][0] = 3; not_sym.data[1][1] = 4;
    ck_assert_int_eq(is_symmetric(not_sym), 0);

    free_matrix(&sym);
    free_matrix(&not_sym);
}
END_TEST

// Test for is_orthogonal
START_TEST(test_is_orthogonal) {
    matrix orth = create_matrix(2, 2);
    orth.data[0][0] = 0; orth.data[0][1] = 1;
    orth.data[1][0] = 1; orth.data[1][1] = 0;
    ck_assert_int_eq(is_orthogonal(orth), 1);

    matrix not_orth = create_matrix(2, 2);
    not_orth.data[0][0] = 1; not_orth.data[0][1] = 2;
    not_orth.data[1][0] = 3; not_orth.data[1][1] = 4;
    ck_assert_int_eq(is_orthogonal(not_orth), 0);

    free_matrix(&orth);
    free_matrix(&not_orth);
}
END_TEST

// Test for is_upper_triangular
START_TEST(test_is_upper_triangular) {
    matrix upper = create_matrix(2, 2);
    upper.data[0][0] = 1; upper.data[0][1] = 2;
    upper.data[1][0] = 0; upper.data[1][1] = 3;
    ck_assert_int_eq(is_upper_triangular(upper), 1);

    matrix not_upper = create_matrix(2, 2);
    not_upper.data[0][0] = 1; not_upper.data[0][1] = 2;
    not_upper.data[1][0] = 3; not_upper.data[1][1] = 4;
    ck_assert_int_eq(is_upper_triangular(not_upper), 0);

    free_matrix(&upper);
    free_matrix(&not_upper);
}
END_TEST

// Test for is_lower_triangular
START_TEST(test_is_lower_triangular) {
    matrix lower = create_matrix(2, 2);
    lower.data[0][0] = 1; lower.data[0][1] = 0;
    lower.data[1][0] = 2; lower.data[1][1] = 3;
    ck_assert_int_eq(is_lower_triangular(lower), 1);

    matrix not_lower = create_matrix(2, 2);
    not_lower.data[0][0] = 1; not_lower.data[0][1] = 2;
    not_lower.data[1][0] = 3; not_lower.data[1][1] = 4;
    ck_assert_int_eq(is_lower_triangular(not_lower), 0);

    free_matrix(&lower);
    free_matrix(&not_lower);
}
END_TEST

// Test for is_identity
START_TEST(test_is_identity) {
    matrix ident = create_matrix(2, 2);
    ident.data[0][0] = 1; ident.data[0][1] = 0;
    ident.data[1][0] = 0; ident.data[1][1] = 1;
    ck_assert_int_eq(is_identity(ident), 1);

    matrix not_ident = create_matrix(2, 2);
    not_ident.data[0][0] = 1; not_ident.data[0][1] = 2;
    not_ident.data[1][0] = 3; not_ident.data[1][1] = 4;
    ck_assert_int_eq(is_identity(not_ident), 0);

    free_matrix(&ident);
    free_matrix(&not_ident);
}
END_TEST

// Test for SVD
START_TEST(test_svd) {
    matrix a = create_matrix(2, 2);
    a.data[0][0] = 1; a.data[0][1] = 2;
    a.data[1][0] = 3; a.data[1][1] = 4;


    matrix U, Sigma, V;

    svd(a, &U, &Sigma, &V);

    matrix temp = multiply_matrices(U, Sigma); 
    matrix Vt = transpose_matrix(V);           
    matrix reconstructed = multiply_matrices(temp, Vt); 


    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            ck_assert_double_eq_tol(reconstructed.data[i][j], a.data[i][j], 1e-3);
        }
    }


    free_matrix(&a);
    free_matrix(&U);
    free_matrix(&Sigma);
    free_matrix(&V);
    free_matrix(&temp);
    free_matrix(&Vt);
    free_matrix(&reconstructed);
}
END_TEST

// Test for Schur decomposition
START_TEST(test_schur_decomposition) {
    matrix a = create_matrix(2, 2);
    a.data[0][0] = 1; a.data[0][1] = 2;
    a.data[1][0] = 3; a.data[1][1] = 4;

    matrix Q, T;
    schur_decomposition(a, &Q, &T, 1000, 1e-6);
    ck_assert_int_eq(is_orthogonal(Q), 1);
    ck_assert_int_eq(is_upper_triangular(T), 1);

    matrix Qt = transpose_matrix(Q);
    matrix temp = multiply_matrices(Q, T);
    matrix reconstructed = multiply_matrices(temp, Qt);

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            ck_assert_double_eq_tol(reconstructed.data[i][j], a.data[i][j], 1e-6);
        }
    }

    free_matrix(&a);
    free_matrix(&Q);
    free_matrix(&T);
    free_matrix(&Qt);
    free_matrix(&temp);
    free_matrix(&reconstructed);
}
END_TEST

// Test for addition of zero matrices
START_TEST(test_add_zero_matrices) {
    matrix a = create_matrix(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.data[i][j] = 0;
        }
    }

    matrix b = create_matrix(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            b.data[i][j] = 0;
        }
    }

    matrix expected = create_matrix(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            expected.data[i][j] = 0;
        }
    }

    matrix result = add_matrices(a, b);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Test for singular matrices multiplication
START_TEST(test_multiply_one_element_matrices) {
    matrix a = create_matrix(1, 1);
    a.data[0][0] = 5;

    matrix b = create_matrix(1, 1);
    b.data[0][0] = 3;

    matrix expected = create_matrix(1, 1);
    expected.data[0][0] = 15;

    matrix result = multiply_matrices(a, b);
    ck_assert_int_eq(matrices_equal(result, expected), 1);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&expected);
    free_matrix(&result);
}
END_TEST

// Test for determinant of zero matrix
START_TEST(test_determinant_zero_matrix) {
    matrix a = create_matrix(2, 2);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            a.data[i][j] = 0;
        }
    }

    double det = determinant(a);
    ck_assert_double_eq_tol(det, 0.0, 1e-6);

    free_matrix(&a);
}
END_TEST

// Test for inverse matrix with determinant 0
START_TEST(test_inverse_singular_matrix) {
    matrix a = create_matrix(2, 2);
    a.data[0][0] = 1; a.data[0][1] = 2;
    a.data[1][0] = 2; a.data[1][1] = 4; 

    double det = determinant(a);
    ck_assert_double_eq_tol(det, 0.0, 1e-6);

    free_matrix(&a);
}
END_TEST

// Create test suite
Suite* matrix_suite(void) {
    Suite *s;
    TCase *tc_core;

    s = suite_create("Matrix");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_add_matrices);
    tcase_add_test(tc_core, test_subtract_matrices);
    tcase_add_test(tc_core, test_multiply_matrices);
    tcase_add_test(tc_core, test_multiply_matrices_non_square);
    tcase_add_test(tc_core, test_multiply_matrices_square_not_power_of_two);
    tcase_add_test(tc_core, test_multiply_matrices_large);
    tcase_add_test(tc_core, test_scalar_multiply);
    tcase_add_test(tc_core, test_transpose_matrix);
    tcase_add_test(tc_core, test_determinant);
    tcase_add_test(tc_core, test_inverse_matrix);
    tcase_add_test(tc_core, test_solve_system);
    tcase_add_test(tc_core, test_rank);
    tcase_add_test(tc_core, test_qr_algorithm);
    tcase_add_test(tc_core, test_generate_random_matrix);
    tcase_add_test(tc_core, test_matrix_power_positive);
    tcase_add_test(tc_core, test_matrix_power_negative);
    tcase_add_test(tc_core, test_cholesky_decomposition);
    tcase_add_test(tc_core, test_lu_decomposition);
    tcase_add_test(tc_core, test_matrix_norms);
    tcase_add_test(tc_core, test_is_diagonal);
    tcase_add_test(tc_core, test_is_symmetric);
    tcase_add_test(tc_core, test_is_orthogonal);
    tcase_add_test(tc_core, test_is_upper_triangular);
    tcase_add_test(tc_core, test_is_lower_triangular);
    tcase_add_test(tc_core, test_is_identity);
    tcase_add_test(tc_core, test_svd);
    tcase_add_test(tc_core, test_schur_decomposition);
    tcase_add_test(tc_core, test_add_zero_matrices);
    tcase_add_test(tc_core, test_multiply_one_element_matrices);
    tcase_add_test(tc_core, test_determinant_zero_matrix);
    tcase_add_test(tc_core, test_inverse_singular_matrix);

    suite_add_tcase(s, tc_core);
    return s;
}

// Main function for test startup
int main(void) {
    int number_failed;
    Suite *s;
    SRunner *sr;
    s = matrix_suite();
    sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}