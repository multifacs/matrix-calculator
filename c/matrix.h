#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>

// Максимальный размер матрицы
#define MAX_SIZE 10

// Структура для матрицы
typedef struct
{
    int rows;
    int cols;
    int data[MAX_SIZE][MAX_SIZE];
} matrix;

// Создание матрицы заданного размера
matrix create_matrix(int rows, int cols);

// Ввод элементов матрицы
void input_matrix(matrix *m);

// Вывод матрицы
void print_matrix(matrix m);

// Сложение двух матриц
matrix add_matrices(matrix a, matrix b);

// Вычитание двух матриц
matrix subtract_matrices(matrix a, matrix b);

// Умножение двух матриц
matrix multiply_matrices(matrix a, matrix b);

// Умножение матрицы на скаляр
matrix scalar_muliply(matrix m, int scalar);

// Транспонирование матрицы
matrix transpose_matrix(matrix m);

// Нахождение минора матрицы
matrix get_minor(matrix m, int row, int col);

// Вычисление определителя матрицы
int determinant(matrix m);

// Прямой ход метода Гаусса (вспомогательная функция)
void gaussian_elimination(matrix *m);

// Обратная матрица (метод Гаусса-Жордана)
matrix inverse_matrix(matrix m);

// Решение системы линейных уравнений Ax = b
matrix solve_system(matrix A, matrix b);

// Нахождение ранга матрицы
int rank(matrix m);

#endif // MATRIX_H
