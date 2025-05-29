#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "constants.h"

// Test for matrix creation and memory allocation
START_TEST(test_create_matrix) {
    matrix m = create_matrix(2, 3);
    ck_assert_int_eq(m.rows, 2);
    ck_assert_int_eq(m.cols, 3);
    ck_assert_ptr_nonnull(m.data);
    for (int i = 0; i < m.rows; i++) {
        ck_assert_ptr_nonnull(m.data[i]);
    }
    free_matrix(&m);
    ck_assert_ptr_null(m.data);
}
END_TEST

// Test for matrix addition
START_TEST(test_add_matrices) {
    matrix a = create_matrix(2, 2);
    a.data[0][0] = 1; a.data[0][1] = 2;
    a.data[1][0] = 3; a.data[1][1] = 4;

    matrix b = create_matrix(2, 2);
    b.data[0][0] = 5; b.data[0][1] = 6;
    b.data[1][0] = 7; b.data[1][1] = 8;

    matrix result;
    int status = add_matrices(a, b, &result);
    ck_assert_int_eq(status, SUCCESS);
    ck_assert_double_eq_tol(result.data[0][0], 6.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[0][1], 8.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[1][0], 10.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[1][1], 12.0, TOLERANCE);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&result);
}
END_TEST

// Test for matrix subtraction
START_TEST(test_subtract_matrices) {
    matrix a = create_matrix(2, 2);
    a.data[0][0] = 5; a.data[0][1] = 6;
    a.data[1][0] = 7; a.data[1][1] = 8;

    matrix b = create_matrix(2, 2);
    b.data[0][0] = 1; b.data[0][1] = 2;
    b.data[1][0] = 3; b.data[1][1] = 4;

    matrix result;
    int status = subtract_matrices(a, b, &result);
    ck_assert_int_eq(status, SUCCESS);
    ck_assert_double_eq_tol(result.data[0][0], 4.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[0][1], 4.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[1][0], 4.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[1][1], 4.0, TOLERANCE);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&result);
}
END_TEST

// Test for matrix multiplication (standard)
START_TEST(test_multiply_matrices) {
    matrix a = create_matrix(2, 2);
    a.data[0][0] = 1; a.data[0][1] = 2;
    a.data[1][0] = 3; a.data[1][1] = 4;

    matrix b = create_matrix(2, 2);
    b.data[0][0] = 5; b.data[0][1] = 6;
    b.data[1][0] = 7; b.data[1][1] = 8;

    matrix result;
    int status = multiply_matrices(a, b, &result);
    ck_assert_int_eq(status, SUCCESS);
    ck_assert_double_eq_tol(result.data[0][0], 19.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[0][1], 22.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[1][0], 43.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[1][1], 50.0, TOLERANCE);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&result);
}
END_TEST

// Test for Strassen matrix multiplication (padded)
START_TEST(test_multiply_matrices_strassen_padded) {
    matrix a = create_matrix(2, 2);
    a.data[0][0] = 1; a.data[0][1] = 2;
    a.data[1][0] = 3; a.data[1][1] = 4;

    matrix b = create_matrix(2, 2);
    b.data[0][0] = 5; b.data[0][1] = 6;
    b.data[1][0] = 7; b.data[1][1] = 8;

    matrix result;
    int status = multiply_matrices_strassen_padded(a, b, &result);
    ck_assert_int_eq(status, SUCCESS);
    ck_assert_double_eq_tol(result.data[0][0], 19.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[0][1], 22.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[1][0], 43.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[1][1], 50.0, TOLERANCE);

    free_matrix(&a);
    free_matrix(&b);
    free_matrix(&result);
}
END_TEST

// Test for determinant calculation
START_TEST(test_determinant) {
    matrix m = create_matrix(2, 2);
    m.data[0][0] = 1; m.data[0][1] = 2;
    m.data[1][0] = 3; m.data[1][1] = 4;

    double det;
    int status = determinant(m, &det);
    ck_assert_int_eq(status, SUCCESS);
    ck_assert_double_eq_tol(det, -2.0, TOLERANCE);

    free_matrix(&m);
}
END_TEST

// Test for scalar multiplication
START_TEST(test_scalar_multiply) {
    matrix m = create_matrix(2, 2);
    m.data[0][0] = 1; m.data[0][1] = 2;
    m.data[1][0] = 3; m.data[1][1] = 4;

    matrix result = scalar_multiply(m, 2.0);
    ck_assert_double_eq_tol(result.data[0][0], 2.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[0][1], 4.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[1][0], 6.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[1][1], 8.0, TOLERANCE);

    free_matrix(&m);
    free_matrix(&result);
}
END_TEST

// Test for matrix transpose
START_TEST(test_transpose_matrix) {
    matrix m = create_matrix(2, 3);
    m.data[0][0] = 1; m.data[0][1] = 2; m.data[0][2] = 3;
    m.data[1][0] = 4; m.data[1][1] = 5; m.data[1][2] = 6;

    matrix result = transpose_matrix(m);
    ck_assert_int_eq(result.rows, 3);
    ck_assert_int_eq(result.cols, 2);
    ck_assert_double_eq_tol(result.data[0][0], 1.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[0][1], 4.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[1][0], 2.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[1][1], 5.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[2][0], 3.0, TOLERANCE);
    ck_assert_double_eq_tol(result.data[2][1], 6.0, TOLERANCE);

    free_matrix(&m);
    free_matrix(&result);
}
END_TEST

// Test for matrix inversion
START_TEST(test_inverse_matrix) {
    // Test with invertible 2x2 matrix
    matrix m = create_matrix(2, 2);
    m.data[0][0] = 4; m.data[0][1] = 7;
    m.data[1][0] = 2; m.data[1][1] = 6;

    matrix inv;
    int status = inverse_matrix(m, &inv);
    ck_assert_int_eq(status, SUCCESS);

    matrix identity;
    int mult_status = multiply_matrices(m, inv, &identity);
    ck_assert_int_eq(mult_status, SUCCESS);
    ck_assert_double_eq_tol(identity.data[0][0], 1.0, TOLERANCE);
    ck_assert_double_eq_tol(identity.data[0][1], 0.0, TOLERANCE);
    ck_assert_double_eq_tol(identity.data[1][0], 0.0, TOLERANCE);
    ck_assert_double_eq_tol(identity.data[1][1], 1.0, TOLERANCE);

    free_matrix(&m);
    free_matrix(&inv);
    free_matrix(&identity);

    // Test with singular matrix
    matrix singular = create_matrix(2, 2);
    singular.data[0][0] = 1; singular.data[0][1] = 2;
    singular.data[1][0] = 2; singular.data[1][1] = 4; // Linearly dependent rows

    matrix inv_singular;
    status = inverse_matrix(singular, &inv_singular);
    ck_assert_int_eq(status, SINGULAR_MATRIX);

    free_matrix(&singular);
}
END_TEST

// Test for solving systems of linear equations
START_TEST(test_solve_system) {
    // Test with square, invertible 2x2 system
    matrix A = create_matrix(2, 2);
    A.data[0][0] = 3; A.data[0][1] = 2;
    A.data[1][0] = 1; A.data[1][1] = 4;

    matrix b = create_matrix(2, 1);
    b.data[0][0] = 7;
    b.data[1][0] = 9;

    matrix x;
    int status = solve_system(A, b, &x);
    ck_assert_int_eq(status, SUCCESS);
    ck_assert_double_eq_tol(x.data[0][0], 1.0, TOLERANCE);
    ck_assert_double_eq_tol(x.data[1][0], 2.0, TOLERANCE);

    free_matrix(&A);
    free_matrix(&b);
    free_matrix(&x);

    // Test with overdetermined system (3x2 matrix)
    matrix A_over = create_matrix(3, 2);
    A_over.data[0][0] = 1; A_over.data[0][1] = 0;
    A_over.data[1][0] = 0; A_over.data[1][1] = 1;
    A_over.data[2][0] = 1; A_over.data[2][1] = 1;

    matrix b_over = create_matrix(3, 1);
    b_over.data[0][0] = 1;
    b_over.data[1][0] = 1;
    b_over.data[2][0] = 1;

    matrix x_over;
    status = solve_system(A_over, b_over, &x_over);
    ck_assert_int_eq(status, SUCCESS);
    ck_assert_double_eq_tol(x_over.data[0][0], 0.6667, 1e-4);
    ck_assert_double_eq_tol(x_over.data[1][0], 0.6667, 1e-4);

    free_matrix(&A_over);
    free_matrix(&b_over);
    free_matrix(&x_over);
}
END_TEST

// Test for matrix rank calculation
START_TEST(test_rank) {
    // Test with full-rank 3x3 matrix
    matrix full_rank = create_matrix(3, 3);
    full_rank.data[0][0] = 1; full_rank.data[0][1] = 0; full_rank.data[0][2] = 0;
    full_rank.data[1][0] = 0; full_rank.data[1][1] = 1; full_rank.data[1][2] = 0;
    full_rank.data[2][0] = 0; full_rank.data[2][1] = 0; full_rank.data[2][2] = 1;

    int r = rank(full_rank);
    ck_assert_int_eq(r, 3);

    free_matrix(&full_rank);

    // Test with rank-deficient 3x3 matrix
    matrix rank_deficient = create_matrix(3, 3);
    rank_deficient.data[0][0] = 1; rank_deficient.data[0][1] = 2; rank_deficient.data[0][2] = 3;
    rank_deficient.data[1][0] = 4; rank_deficient.data[1][1] = 5; rank_deficient.data[1][2] = 6;
    rank_deficient.data[2][0] = 7; rank_deficient.data[2][1] = 8; rank_deficient.data[2][2] = 9;

    r = rank(rank_deficient);
    ck_assert_int_eq(r, 2);

    free_matrix(&rank_deficient);

    // Test with zero 2x2 matrix
    matrix zero = create_matrix(2, 2);
    zero.data[0][0] = 0; zero.data[0][1] = 0;
    zero.data[1][0] = 0; zero.data[1][1] = 0;

    r = rank(zero);
    ck_assert_int_eq(r, 0);

    free_matrix(&zero);
}
END_TEST

// Test for matrix properties
START_TEST(test_matrix_properties) {
    // Test diagonal matrix
    matrix diag = create_matrix(2, 2);
    diag.data[0][0] = 1; diag.data[0][1] = 0;
    diag.data[1][0] = 0; diag.data[1][1] = 2;
    ck_assert_int_eq(is_diagonal(diag), 1);
    ck_assert_int_eq(is_upper_triangular(diag), 1);
    ck_assert_int_eq(is_lower_triangular(diag), 1);
    ck_assert_int_eq(is_symmetric(diag), 1);
    free_matrix(&diag);

    // Test non-diagonal matrix
    matrix non_diag = create_matrix(2, 2);
    non_diag.data[0][0] = 1; non_diag.data[0][1] = 3;
    non_diag.data[1][0] = 0; non_diag.data[1][1] = 2;
    ck_assert_int_eq(is_diagonal(non_diag), 0);
    ck_assert_int_eq(is_upper_triangular(non_diag), 1);
    ck_assert_int_eq(is_lower_triangular(non_diag), 0);
    ck_assert_int_eq(is_symmetric(non_diag), 0);
    free_matrix(&non_diag);

    // Test identity matrix
    matrix identity = create_identity_matrix(2);
    ck_assert_int_eq(is_identity(identity), 1);
    ck_assert_int_eq(is_diagonal(identity), 1);
    ck_assert_int_eq(is_orthogonal(identity), 1);
    free_matrix(&identity);

    // Test orthogonal matrix (rotation matrix)
    matrix rot = create_matrix(2, 2);
    rot.data[0][0] = 0; rot.data[0][1] = -1;
    rot.data[1][0] = 1; rot.data[1][1] = 0;
    ck_assert_int_eq(is_orthogonal(rot), 1);
    free_matrix(&rot);
}
END_TEST

// Test for matrix norms
START_TEST(test_matrix_norms) {
    matrix m = create_matrix(2, 2);
    m.data[0][0] = 1; m.data[0][1] = 2;
    m.data[1][0] = 3; m.data[1][1] = 4;

    double f_norm = frobenius_norm(m);
    ck_assert_double_eq_tol(f_norm, sqrt(30), TOLERANCE);

    double one_norm_val = one_norm(m);
    ck_assert_double_eq_tol(one_norm_val, 6, TOLERANCE);

    double inf_norm = infinity_norm(m);
    ck_assert_double_eq_tol(inf_norm, 7, TOLERANCE);

    free_matrix(&m);
}
END_TEST

// Test for Cholesky decomposition
START_TEST(test_cholesky_decomposition) {
    // Test with positive definite matrix
    matrix m = create_matrix(2, 2);
    m.data[0][0] = 4; m.data[0][1] = 1;
    m.data[1][0] = 1; m.data[1][1] = 1;

    matrix L;
    int status = cholesky_decomposition(m, &L);
    ck_assert_int_eq(status, SUCCESS);
    ck_assert_double_eq_tol(L.data[0][0], 2, TOLERANCE);
    ck_assert_double_eq_tol(L.data[0][1], 0, TOLERANCE);
    ck_assert_double_eq_tol(L.data[1][0], 0.5, TOLERANCE);
    ck_assert_double_eq_tol(L.data[1][1], sqrt(0.75), TOLERANCE);

    free_matrix(&m);
    free_matrix(&L);

    // Test with non-positive definite matrix
    matrix non_pd = create_matrix(2, 2);
    non_pd.data[0][0] = 1; non_pd.data[0][1] = 2;
    non_pd.data[1][0] = 2; non_pd.data[1][1] = 1;
    status = cholesky_decomposition(non_pd, &L);
    ck_assert_int_eq(status, NOT_POSITIVE_DEFINITE);
    free_matrix(&non_pd);
}
END_TEST

// Test for LU decomposition
START_TEST(test_lu_decomposition) {
    matrix m = create_matrix(2, 2);
    m.data[0][0] = 2; m.data[0][1] = 1;
    m.data[1][0] = 1; m.data[1][1] = 2;

    matrix L, U;
    int status = lu_decomposition(m, &L, &U);
    ck_assert_int_eq(status, SUCCESS);

    // Verify L is lower triangular with ones on diagonal
    ck_assert_double_eq_tol(L.data[0][0], 1, TOLERANCE);
    ck_assert_double_eq_tol(L.data[0][1], 0, TOLERANCE);
    ck_assert_double_eq_tol(L.data[1][0], 0.5, TOLERANCE);
    ck_assert_double_eq_tol(L.data[1][1], 1, TOLERANCE);

    // Verify U is upper triangular
    ck_assert_double_eq_tol(U.data[0][0], 2, TOLERANCE);
    ck_assert_double_eq_tol(U.data[0][1], 1, TOLERANCE);
    ck_assert_double_eq_tol(U.data[1][0], 0, TOLERANCE);
    ck_assert_double_eq_tol(U.data[1][1], 1.5, TOLERANCE);

    free_matrix(&m);
    free_matrix(&L);
    free_matrix(&U);

    // Test with singular matrix
    matrix singular = create_matrix(2, 2);
    singular.data[0][0] = 1; singular.data[0][1] = 2;
    singular.data[1][0] = 2; singular.data[1][1] = 4;
    status = lu_decomposition(singular, &L, &U);
    ck_assert_int_eq(status, SINGULAR_MATRIX);
    free_matrix(&singular);
}
END_TEST

// Test for special matrix generation
START_TEST(test_special_matrices) {
    // Test Hilbert matrix
    matrix hilbert;
    int status = generate_hilbert_matrix(2, &hilbert);
    ck_assert_int_eq(status, SUCCESS);
    ck_assert_double_eq_tol(hilbert.data[0][0], 1.0, TOLERANCE);
    ck_assert_double_eq_tol(hilbert.data[0][1], 0.5, TOLERANCE);
    ck_assert_double_eq_tol(hilbert.data[1][0], 0.5, TOLERANCE);
    ck_assert_double_eq_tol(hilbert.data[1][1], 1.0 / 3, TOLERANCE);
    free_matrix(&hilbert);

    // Test Hadamard matrix
    matrix hadamard;
    status = generate_hadamard_matrix(2, &hadamard);
    ck_assert_int_eq(status, SUCCESS);
    ck_assert_double_eq_tol(hadamard.data[0][0], 1, TOLERANCE);
    ck_assert_double_eq_tol(hadamard.data[0][1], 1, TOLERANCE);
    ck_assert_double_eq_tol(hadamard.data[1][0], 1, TOLERANCE);
    ck_assert_double_eq_tol(hadamard.data[1][1], -1, TOLERANCE);
    free_matrix(&hadamard);

    // Test invalid Hadamard matrix size
    matrix invalid_hadamard;
    status = generate_hadamard_matrix(3, &invalid_hadamard);
    ck_assert_int_eq(status, INVALID_INPUT);
}
END_TEST

// Create test suite
Suite* matrix_suite(void) {
    Suite *s = suite_create("Matrix Calculator Tests");
    TCase *tc_core = tcase_create("Core Operations");

    tcase_add_test(tc_core, test_create_matrix);
    tcase_add_test(tc_core, test_add_matrices);
    tcase_add_test(tc_core, test_subtract_matrices);
    tcase_add_test(tc_core, test_multiply_matrices);
    tcase_add_test(tc_core, test_multiply_matrices_strassen_padded);
    tcase_add_test(tc_core, test_determinant);
    tcase_add_test(tc_core, test_scalar_multiply);
    tcase_add_test(tc_core, test_transpose_matrix);
    tcase_add_test(tc_core, test_inverse_matrix);
    tcase_add_test(tc_core, test_solve_system);
    tcase_add_test(tc_core, test_rank);
    tcase_add_test(tc_core, test_matrix_properties);
    tcase_add_test(tc_core, test_matrix_norms);
    tcase_add_test(tc_core, test_cholesky_decomposition);
    tcase_add_test(tc_core, test_lu_decomposition);
    tcase_add_test(tc_core, test_special_matrices);

    suite_add_tcase(s, tc_core);
    return s;
}

// Main function to run tests
int main(void) {
    int number_failed;
    Suite *s = matrix_suite();
    SRunner *sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}