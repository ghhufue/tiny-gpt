#include "matrix.h"

#include <algorithm>
#include <cassert>
#include <cmath>

Matrix::Matrix(int rows, int cols)
    : rows(rows), cols(cols), data(rows * cols, 0.0)
{
    assert(rows > 0);
    assert(cols > 0);
}

Matrix::Matrix(int rows, int cols, double value)
    : rows(rows), cols(cols), data(rows * cols, value)
{
    assert(rows > 0);
    assert(cols > 0);
}

double& Matrix::operator()(int r, int c)
{
    assert(r >= 0);
    assert(r < rows);
    assert(c >= 0);
    assert(c < cols);
    return data[r * cols + c];
}

double Matrix::operator()(int r, int c) const
{
    assert(r >= 0);
    assert(r < rows);
    assert(c >= 0);
    assert(c < cols);
    return data[r * cols + c];
}

Matrix Matrix::transpose() const
{
    Matrix result(cols, rows);
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            result(c, r) = (*this)(r, c);
        }
    }
    return result;
}

Matrix Matrix::matmul(const Matrix& other) const
{
    assert(cols == other.rows);

    Matrix result(rows, other.cols);
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < other.cols; c++)
        {
            double sum = 0.0;
            for (int k = 0; k < cols; k++)
            {
                sum += (*this)(r, k) * other(k, c);
            }
            result(r, c) = sum;
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix& other) const
{
    assert(rows == other.rows);
    assert(cols == other.cols);

    Matrix result(rows, cols);
    for (int i = 0; i < static_cast<int>(data.size()); i++)
    {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Matrix Matrix::add_row_vector(const std::vector<double>& bias) const
{
    assert(static_cast<int>(bias.size()) == cols);

    Matrix result(rows, cols);
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            result(r, c) = (*this)(r, c) + bias[c];
        }
    }
    return result;
}

Matrix Matrix::relu() const
{
    Matrix result(rows, cols);
    for (int i = 0; i < static_cast<int>(data.size()); i++)
    {
        result.data[i] = std::max(0.0, data[i]);
    }
    return result;
}

Matrix Matrix::softmax_rows() const
{
    Matrix result(rows, cols);
    for (int r = 0; r < rows; r++)
    {
        double max_value = (*this)(r, 0);
        for (int c = 1; c < cols; c++)
        {
            max_value = std::max(max_value, (*this)(r, c));
        }

        double sum = 0.0;
        for (int c = 0; c < cols; c++)
        {
            double value = std::exp((*this)(r, c) - max_value);
            result(r, c) = value;
            sum += value;
        }

        assert(sum > 0.0);
        for (int c = 0; c < cols; c++)
        {
            result(r, c) /= sum;
        }
    }
    return result;
}
