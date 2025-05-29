#ifndef UTILS_H
#define UTILS_H

#include "matrix_lib/matrix.h"

void save_matrix_to_file(matrix m, const char* filename);
matrix load_matrix_from_file(const char* filename);

#endif // UTILS_H