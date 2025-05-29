#include "ui.h"
#include <stdio.h>
#include <stdlib.h>
#include "matrix_lib/matrix.h" 
#include "constants.h"

/**
 * @file ui.c
 * @brief User interface functions for the matrix calculator.
 *
 * This file contains functions for interacting with the user, including displaying menus,
 * getting input, and displaying results. It uses ANSI escape codes for colored output
 * to enhance the user experience.
 */

/**
 * @brief Displays the main menu of the matrix calculator.
 *
 * Prints a list of available operations to the console, each with a corresponding number.
 * Uses colored text for better readability.
 */
void show_menu() {
    printf("\n%sMatrix Calculator Menu:%s\n\n", UCYN, COLOR_RESET); 
    printf("%s1. Add matrices%s\n", UYEL, COLOR_RESET);
    printf("%s2. Subtract matrices%s\n", UYEL, COLOR_RESET);
    printf("%s3. Multiply matrices%s\n", UYEL, COLOR_RESET);
    printf("%s4. Scalar multiplication%s\n", UYEL, COLOR_RESET);
    printf("%s5. Transpose matrix%s\n", UYEL, COLOR_RESET);
    printf("%s6. Determinant%s\n", UYEL, COLOR_RESET);
    printf("%s7. Inverse matrix%s\n", UYEL, COLOR_RESET);
    printf("%s8. Solve system%s\n", UYEL, COLOR_RESET);
    printf("%s9. Rank%s\n", UYEL, COLOR_RESET);
    printf("%s10. Generate random matrix%s\n", UYEL, COLOR_RESET);
    printf("%s11. Check properties%s\n", UYEL, COLOR_RESET);
    printf("%s12. Matrix power%s\n", UYEL, COLOR_RESET);
    printf("%s13. Cholesky decomposition%s\n", UYEL, COLOR_RESET);
    printf("%s14. Eigenvalues and eigenvectors%s\n", UYEL, COLOR_RESET);
    printf("%s15. LU decomposition%s\n", UYEL, COLOR_RESET);
    printf("%s16. Matrix norms%s\n", UYEL, COLOR_RESET);
    printf("%s17. Save matrix%s\n", UYEL, COLOR_RESET);
    printf("%s18. Load matrix%s\n", UYEL, COLOR_RESET);
    printf("%s19. Use loaded matrix%s\n", UYEL, COLOR_RESET);
    printf("%s20. Save random matrix%s\n", UYEL, COLOR_RESET);
    printf("%s21. SVD%s\n", UYEL, COLOR_RESET);
    printf("%s22. Schur decomposition%s\n", UYEL, COLOR_RESET);
    printf("%s23. Generate special matrix%s\n", UYEL, COLOR_RESET);
    printf("%s24. Hessenberg form%s\n", UYEL, COLOR_RESET);
    printf("%s25. Exit%s\n", UYEL, COLOR_RESET);
    printf("%sEnter your choice: %s", UBLU, COLOR_RESET); 
}

/**
 * @brief Retrieves the user's choice from the menu.
 *
 * Reads an integer input from the user and clears the input buffer.
 *
 * @return int The user's choice.
 */
int get_user_choice() {
    int choice;
    scanf("%d", &choice);
    while (getchar() != '\n'); 
    return choice;
}

/**
 * @brief Prompts the user to input a positive integer.
 *
 * Displays the provided prompt and reads an integer from the user. If the input is not
 * a positive integer, it displays an error message and prompts again.
 *
 * @param prompt The message to display to the user.
 * @return int The positive integer entered by the user.
 */
int input_positive_integer(const char* prompt) {
    int value;
    do {
        printf("%s%s%s", UBLU, prompt, COLOR_RESET);
        scanf("%d", &value);
        while (getchar() != '\n');
        if (value <= 0) {
            printf("%sError: Please enter a positive integer.\n%s", URED, COLOR_RESET);
        }
    } while (value <= 0);
    return value;
}

/**
 * @brief Collects parameters for generating a random matrix.
 *
 * Prompts the user to enter the number of rows, columns, minimum value, and maximum value
 * for the random matrix elements.
 *
 * @param rows Pointer to store the number of rows.
 * @param cols Pointer to store the number of columns.
 * @param min_val Pointer to store the minimum value.
 * @param max_val Pointer to store the maximum value.
 */
void input_random_matrix_params(int *rows, int *cols, double *min_val, double *max_val) {
    *rows = input_positive_integer("Enter number of rows: ");
    *cols = input_positive_integer("Enter number of columns: ");
    printf("%sEnter minimum value: %s", UBLU, COLOR_RESET);
    scanf("%lf", min_val);
    printf("%sEnter maximum value: %s", UBLU, COLOR_RESET);
    scanf("%lf", max_val);
    while (getchar() != '\n');
}

/**
 * @brief Creates a new matrix by getting dimensions and elements from the user.
 *
 * Asks the user for the number of rows and columns, creates a matrix of that size,
 * prompts the user to input the elements, and allows editing of the matrix.
 *
 * @return matrix The newly created and populated matrix.
 */
matrix input_matrix_new() {
    int rows = input_positive_integer("Enter number of rows: ");
    int cols = input_positive_integer("Enter number of columns: ");
    matrix m = create_matrix(rows, cols);
    if (m.data == NULL) {
        printf("%sError: Failed to allocate matrix.\n%s", URED, COLOR_RESET);
        return m;
    }
    printf("%sEnter matrix elements:\n%s", UBLU, COLOR_RESET);
    input_matrix(&m);  
    edit_matrix(&m);
    return m;
}

/**
 * @brief Displays the given matrix.
 *
 * Checks if the matrix data is valid and, if so, prints the matrix using the
 * `print_matrix` function from the matrix library.
 *
 * @param m The matrix to display.
 */
void display_matrix(matrix m) {
    if (m.data == NULL) {
        printf("%sError: Cannot display null matrix.\n%s", URED, COLOR_RESET);
        return;
    }
    print_matrix(m);  
}

/**
 * @brief Retrieves a scalar value from the user.
 *
 * Prompts the user to enter a scalar value and reads it from input.
 *
 * @return double The scalar value entered by the user.
 */
double input_scalar() {
    double scalar;
    printf("%sEnter scalar value: %s", UBLU, COLOR_RESET);
    scanf("%lf", &scalar);
    while (getchar() != '\n');
    return scalar;
}

/**
 * @brief Pauses execution until the user presses Enter.
 *
 * Displays a message asking the user to press Enter and waits for the input.
 */
void wait_for_enter() {
    printf("%sPress Enter to continue...%s", UYEL, COLOR_RESET);
    while (getchar() != '\n');
}

/**
 * @brief Displays the eigenvalues and eigenvectors of a matrix.
 *
 * Prints headers for eigenvalues and eigenvectors and displays the respective matrices.
 *
 * @param eigenvalues Matrix containing the eigenvalues.
 * @param eigenvectors Matrix containing the eigenvectors.
 */
void display_eigen(matrix eigenvalues, matrix eigenvectors) {
    printf("\n%sEigenvalues:%s\n", UGRN, COLOR_RESET);
    display_matrix(eigenvalues);
    printf("\n%sEigenvectors:%s\n", UGRN, COLOR_RESET);
    display_matrix(eigenvectors);
}