#pragma once

#include <vector>

class Matrix
{
public:
    int rows;
    int cols;
    std::vector<double> data;

    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double value);

    double& operator()(int r, int c);
    double operator()(int r, int c) const;

    Matrix transpose() const;
    Matrix matmul(const Matrix& other) const;
    Matrix add(const Matrix& other) const;
    Matrix add_row_vector(const std::vector<double>& bias) const;

    Matrix relu() const;
    Matrix softmax_rows() const;
};
