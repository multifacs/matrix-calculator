#ifndef UI_H
#define UI_H

#include "matrix_lib/matrix.h"

void show_menu();
int get_user_choice();
int input_positive_integer(const char* prompt);
void input_random_matrix_params(int *rows, int *cols, double *min_val, double *max_val);
matrix input_matrix_new();
void display_matrix(matrix m);
double input_scalar();
void wait_for_enter();
void display_eigen(matrix eigenvalues, matrix eigenvectors);

#endif // UI_H