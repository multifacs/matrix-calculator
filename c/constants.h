#ifndef CONSTANTS_H
#define CONSTANTS_H

/**
 * @def TOLERANCE
 * @brief Tolerance for floating-point comparisons.
 *
 * This value is used to determine when two floating-point numbers are considered equal,
 * accounting for precision errors in numerical computations.
 */
#define TOLERANCE               1e-9

/**
 * @def DISPLAY_TOL
 * @brief Tolerance for displaying values as zero.
 *
 * If the absolute value of a matrix element is less than this tolerance, it is displayed as zero.
 * This helps in cleaning up the output of matrices with very small values due to floating-point errors.
 */
#define DISPLAY_TOL             1e-3

/**
 * @def STRASSEN_THRESHOLD
 * @brief Threshold for switching to standard matrix multiplication in Strassen's algorithm.
 *
 * For matrices smaller than this size, the standard matrix multiplication is used instead of
 * Strassen's algorithm to optimize performance.
 */
#define STRASSEN_THRESHOLD     64


// Color codes for console output
#define URED                    "\e[4;31m"      // Red underlined for errors
#define UGRN                    "\e[4;32m"      // Green underlined for results
#define UYEL                    "\e[4;33m"      // Yellow underlined for information/warnings
#define UBLU                    "\e[4;34m"      // Blue underlined for inputs
#define UCYN                    "\e[4;36m"      // Cyan underlined for menus
#define COLOR_RESET             "\e[0m"         // Reset color formatting

/**
 * @enum ErrorCode
 * @brief Enumeration of error codes for matrix operations.
 *
 * These codes are used to indicate the success or specific failure reasons of matrix functions,
 * allowing for consistent error handling across the library.
 */
enum ErrorCode {
    SUCCESS = 0,
    INVALID_DIMENSIONS = 1,
    SINGULAR_MATRIX = 2,
    INVALID_INPUT = 3,
    NOT_SYMMETRIC = 4,
    NOT_POSITIVE_DEFINITE = 5
};

#endif // CONSTANTS_H