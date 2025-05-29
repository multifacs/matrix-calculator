#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ui.h"
#include "operations.h"
#include "constants.h"


/**
 * @file main.c
 * @brief Main entry point for the matrix operations program.
 *
 * This file contains the main function that drives the program. It initializes the random number generator,
 * displays the menu, and handles user input to perform various matrix operations. It also manages the lifecycle
 * of a saved matrix that can be loaded and used across operations.
 */

/**
 * @brief Function pointer type for matrix operations.
 *
 * This type defines a function pointer that takes a pointer to a matrix and a pointer to an integer
 * (indicating whether a matrix is loaded) and returns an integer status code.
 */
typedef int (*operation_func)(matrix *, int *);

/**
 * @brief Array of function pointers for matrix operations.
 *
 * Each index corresponds to a menu choice, mapping to the respective operation function.
 * Index 0 is NULL as menu choices start from 1.
 */
operation_func operations[] = {
    NULL, 
    add_matrices_operation,
    subtract_matrices_operation,
    multiply_matrices_operation,
    scalar_multiply_operation,
    transpose_operation,
    determinant_operation,
    inverse_operation,
    solve_system_operation,
    rank_operation,
    generate_random_matrix_operation,
    check_properties_operation,
    matrix_power_operation,
    cholesky_decomposition_operation,
    eigenvalues_operation,
    lu_decomposition_operation,
    matrix_norms_session_operation,
    save_matrix_operation,
    load_matrix_operation,
    use_loaded_matrix_operation,
    save_random_matrix_operation,
    svd_operation,
    schur_decomposition_operation,
    generate_special_matrix_operation,
    hessenberg_form_operation,
    exit_operation
};

/**
 * @brief Main function to run the matrix operations program.
 *
 * Initializes the random number generator, manages the main loop for user interaction,
 * and handles the execution of selected matrix operations. It also ensures that any
 * loaded matrix is properly freed before exiting.
 *
 * @return int Exit status of the program (0 for success).
 */
int main() {
    srand(time(NULL));  // Seed the random number generator with current time
    int choice;
    matrix saved_matrix = {0, 0, NULL}; // Initialize an empty matrix for saving/loading
    int matrix_loaded = 0;  // Flag to indicate if a matrix is currently loaded

    do {
        show_menu();    // Display the menu of available operations
        choice = get_user_choice(); // Get the user's selection
        if (choice >= 1 && choice <= 25) {
            // Execute the selected operation and check for errors
            int error = operations[choice](&saved_matrix, &matrix_loaded);
            if (error && choice != 25) {
                // Print error message if operation fails (except for exit)
                printf("%sOperation failed with error code %d.\n%s", URED, error, COLOR_RESET);
            }
        } else {
            // Handle invalid menu choices
            printf("\n%sWrong choice. Please enter a number between 1 and 25.\n%s", URED, COLOR_RESET);
        }
    } while (choice != 25); // Continue until the user chooses to exit

    // Free the loaded matrix if it was allocated
    if (matrix_loaded && saved_matrix.data != NULL) {
        free_matrix(&saved_matrix);
    }

    return 0;
}