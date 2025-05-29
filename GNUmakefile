# Specify path to C source files
C_SRCS := c/main.c c/ui.c c/utils.c c/operations.c c/matrix_lib/matrix_core.c c/matrix_lib/matrix_decomposition.c c/matrix_lib/matrix_properties.c c/matrix_lib/matrix_advanced.c c/matrix_lib/matrix_special.c

# Specify path to Rust source file
RUST_SRC := rust/main.rs

# Common build directory for binaries
BUILD_DIR := build

# Path to C output file
C_OUT := $(BUILD_DIR)/c/hello_c

# Path to Rust output file
RUST_OUT := $(BUILD_DIR)/rs/hello_rs

# Path to test source file
TEST_SRC := tests/matrix_c/test_matrix.c

# Path to test output file
TEST_OUT := $(BUILD_DIR)/test_matrix

# Specify compilers
CC := gcc
RUSTC := rustc

# Compilation flags
CFLAGS := -I c/ -I c/matrix_lib -lm -lcheck

# .PHONY indicates targets that do not correspond to actual files
.PHONY: all clean c rust test

## @brief Default target: builds both C and Rust programs
all: c rust

# ---- C BUILD ----

## @brief Target to build the C program
## @depends $(C_OUT) The compiled C output file
c: $(C_OUT)

## @brief Rule to compile C source into an executable
$(C_OUT): $(C_SRCS)
	# Create build directory if it doesn't exist
	@mkdir -p $(dir $@)
	# Compile all C source files into hello_c
	$(CC) $(C_SRCS) -o $@ $(CFLAGS)

## @brief Rule to compile test source file
$(TEST_OUT): $(TEST_SRC) $(filter-out c/main.c, $(C_SRCS))
	@mkdir -p $(BUILD_DIR)
	$(CC) -I c/ -I c/matrix_lib -o $@ $(TEST_SRC) $(filter-out c/main.c, $(C_SRCS)) -lcheck -lm

## @brief Target to build and run tests
test: $(TEST_OUT)
	./$(TEST_OUT)

# ---- RUST BUILD ----

## @brief Target to build the Rust program
rust: $(RUST_OUT)

## @brief Rule to compile Rust source into an executable
$(RUST_OUT): $(RUST_SRC)
	# Create build directory if it doesn't exist
	@mkdir -p $(dir $@)
	# Compile main.rs into hello_rs
	$(RUSTC) $< -o $@

# ---- CLEAN ----

## @brief Target to clean up the build directory
clean:
	rm -rf $(BUILD_DIR)