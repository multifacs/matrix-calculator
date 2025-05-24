# GNUmakefile — сборка hello world на C и Rust с размещением в отдельных папках

# Указываем путь к C-исходнику
C_SRC := c/main.c

# Указываем путь к Rust-исходнику
RUST_SRC := rust/main.rs

# Общая папка, куда кладём бинарники
BUILD_DIR := build

# Путь к выходному файлу на C
C_OUT := $(BUILD_DIR)/c/hello_c

# Путь к выходному файлу на Rust
RUST_OUT := $(BUILD_DIR)/rs/hello_rs

# Указываем компиляторы. Можно переопределить через командную строку:
# make CC=clang RUSTC=rustc
CC := gcc
RUSTC := rustc

# .PHONY означает, что цели не соответствуют настоящим файлам — они всегда будут выполняться, если указаны.
.PHONY: all clean c rust

# Цель по умолчанию — собираем и C, и Rust
all: c rust

# ---- C BUILD ----

# Цель для сборки C-программы
# Зависит от результата сборки: $(C_OUT)
c: $(C_OUT)

# Правило сборки исполняемого файла из C-кода
$(C_OUT): $(C_SRC)
	# Создаём папку, если её ещё нет
	@mkdir -p $(dir $@)

	# Компилируем main.c → hello_c
	# $< — первый файл зависимости (c/main.c)
	# $@ — цель (build/c/hello_c)
	$(CC) $< -o $@

# ---- RUST BUILD ----

# Цель для сборки Rust-программы
rust: $(RUST_OUT)

# Правило сборки исполняемого файла из Rust-кода
$(RUST_OUT): $(RUST_SRC)
	# Создаём папку, если её ещё нет
	@mkdir -p $(dir $@)

	# Компилируем main.rs → hello_rs
	# $< — rust/main.rs
	# $@ — build/rs/hello_rs
	$(RUSTC) $< -o $@

# ---- CLEAN ----

# Цель очистки — удаляет всю папку сборки
clean:
	rm -rf $(BUILD_DIR)
