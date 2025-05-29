#include <stdio.h>
#include "utils.h"
#include "constants.h"

/**
 * @file utils.c
 * @brief Utility functions for saving and loading matrices to/from files.
 *
 * This file contains functions to save a matrix to a file and load a matrix from a file.
 * It uses colored console output for user feedback and handles various error conditions.
 */

/**
 * @brief Saves a matrix to a file.
 *
 * This function writes the dimensions and elements of the matrix to the specified file.
 * If the file cannot be opened, it prints an error message. On success, it confirms
 * the save operation with a message.
 *
 * @param m The matrix to save.
 * @param filename The name of the file to save the matrix to.
 */
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

/**
 * @brief Loads a matrix from a file.
 *
 * This function reads the dimensions and elements of a matrix from the specified file.
 * It performs checks to ensure the file exists, is correctly formatted, and contains
 * valid matrix data. If any errors occur during loading, it prints an error message
 * and returns an empty matrix.
 *
 * @param filename The name of the file to load the matrix from.
 * @return matrix The loaded matrix, or an empty matrix if loading fails.
 */
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