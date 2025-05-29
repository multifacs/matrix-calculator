#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "matrix_lib/matrix.h"

int add_matrices_operation(matrix *saved_matrix, int *matrix_loaded);
int subtract_matrices_operation(matrix *saved_matrix, int *matrix_loaded);
int multiply_matrices_operation(matrix *saved_matrix, int *matrix_loaded);
int scalar_multiply_operation(matrix *saved_matrix, int *matrix_loaded);
int transpose_operation(matrix *saved_matrix, int *matrix_loaded);
int determinant_operation(matrix *saved_matrix, int *matrix_loaded);
int inverse_operation(matrix *saved_matrix, int *matrix_loaded);
int solve_system_operation(matrix *saved_matrix, int *matrix_loaded);
int rank_operation(matrix *saved_matrix, int *matrix_loaded);
int generate_random_matrix_operation(matrix *saved_matrix, int *matrix_loaded);
int check_properties_operation(matrix *saved_matrix, int *matrix_loaded);
int matrix_power_operation(matrix *saved_matrix, int *matrix_loaded);
int cholesky_decomposition_operation(matrix *saved_matrix, int *matrix_loaded);
int eigenvalues_operation(matrix *saved_matrix, int *matrix_loaded);
int lu_decomposition_operation(matrix *saved_matrix, int *matrix_loaded);
int matrix_norms_session_operation(matrix *saved_matrix, int *matrix_loaded);
int save_matrix_operation(matrix *saved_matrix, int *matrix_loaded);
int load_matrix_operation(matrix *saved_matrix, int *matrix_loaded);
int use_loaded_matrix_operation(matrix *saved_matrix, int *matrix_loaded);
int save_random_matrix_operation(matrix *saved_matrix, int *matrix_loaded);
int svd_operation(matrix *saved_matrix, int *matrix_loaded);
int schur_decomposition_operation(matrix *saved_matrix, int *matrix_loaded);
int generate_special_matrix_operation(matrix *saved_matrix, int *matrix_loaded);
int hessenberg_form_operation(matrix *saved_matrix, int *matrix_loaded);
int exit_operation(matrix *saved_matrix, int *matrix_loaded);

#endif // OPERATIONS_H