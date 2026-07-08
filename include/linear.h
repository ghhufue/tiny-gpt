#pragma once

#include <vector>

#include "matrix.h"

class Linear
{
public:
    int in_features;
    int out_features;

    Matrix weight; // shape: [in_features, out_features]
    std::vector<double> bias;

    Matrix d_weight; // shape: [in_features, out_features]
    std::vector<double> d_bias;

    Linear(int in_features, int out_features,
           double weight_init_scale = 0.01, unsigned int random_seed = 42);

    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output);
    void step(double learning_rate);
    void zero_grad();

private:
    Matrix input_cache;
    bool has_input_cache;
};
