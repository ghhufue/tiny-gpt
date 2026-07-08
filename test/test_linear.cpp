#include <cassert>
#include <cmath>
#include <iostream>

#include "../include/linear.h"

namespace
{
bool near(double a, double b)
{
    return std::abs(a - b) < 1e-9;
}
}

int main()
{
    Linear linear(3, 2);

    linear.weight(0, 0) = 1.0;
    linear.weight(0, 1) = 2.0;
    linear.weight(1, 0) = 3.0;
    linear.weight(1, 1) = 4.0;
    linear.weight(2, 0) = 5.0;
    linear.weight(2, 1) = 6.0;
    linear.bias[0] = 0.5;
    linear.bias[1] = -0.5;

    Matrix input(2, 3);
    input(0, 0) = 1.0;
    input(0, 1) = 2.0;
    input(0, 2) = 3.0;
    input(1, 0) = 4.0;
    input(1, 1) = 5.0;
    input(1, 2) = 6.0;

    Matrix output = linear.forward(input);
    assert(output.rows == 2);
    assert(output.cols == 2);
    assert(near(output(0, 0), 22.5));
    assert(near(output(0, 1), 27.5));
    assert(near(output(1, 0), 49.5));
    assert(near(output(1, 1), 63.5));

    Matrix grad_output(2, 2);
    grad_output(0, 0) = 1.0;
    grad_output(0, 1) = 2.0;
    grad_output(1, 0) = 3.0;
    grad_output(1, 1) = 4.0;

    Matrix grad_input = linear.backward(grad_output);
    assert(grad_input.rows == 2);
    assert(grad_input.cols == 3);
    assert(near(grad_input(0, 0), 5.0));
    assert(near(grad_input(0, 1), 11.0));
    assert(near(grad_input(0, 2), 17.0));
    assert(near(grad_input(1, 0), 11.0));
    assert(near(grad_input(1, 1), 25.0));
    assert(near(grad_input(1, 2), 39.0));

    assert(near(linear.d_weight(0, 0), 13.0));
    assert(near(linear.d_weight(0, 1), 18.0));
    assert(near(linear.d_weight(1, 0), 17.0));
    assert(near(linear.d_weight(1, 1), 24.0));
    assert(near(linear.d_weight(2, 0), 21.0));
    assert(near(linear.d_weight(2, 1), 30.0));
    assert(near(linear.d_bias[0], 4.0));
    assert(near(linear.d_bias[1], 6.0));

    linear.step(0.1);
    assert(near(linear.weight(0, 0), -0.3));
    assert(near(linear.weight(2, 1), 3.0));
    assert(near(linear.bias[0], 0.1));
    assert(near(linear.bias[1], -1.1));
    assert(near(linear.d_weight(0, 0), 0.0));
    assert(near(linear.d_bias[0], 0.0));

    std::cout << "All linear tests passed!" << std::endl;

    return 0;
}
