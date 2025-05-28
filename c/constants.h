#ifndef CONSTANTS_H
#define CONSTANTS_H

#define TOLERANCE               1e-9           // Tolerance for floating-point comparisons
#define DISPLAY_TOL             1e-3           // Updated tolerance for zero display
#define STRASSEN_THRESHOLD     64             // Determines when algorithm switches to standart multiplication

#define URED                    "\e[4;31m"      // Red underlined for errors
#define UGRN                    "\e[4;32m"      // Green underlined for results
#define UYEL                    "\e[4;33m"      // Yellow underlined for information/warnings
#define UBLU                    "\e[4;34m"      // Blue underlined for inputs
#define UCYN                    "\e[4;36m"      // Cyan underlined for menus
#define COLOR_RESET             "\e[0m"         // Reset color formatting

#endif // CONSTANTS_H