#include "linear.h"

#include <cassert>
#include <random>

Linear::Linear(int in_features, int out_features,
               double weight_init_scale, unsigned int random_seed)
    : in_features(in_features),
      out_features(out_features),
      weight(in_features, out_features),
      bias(out_features, 0.0),
      d_weight(in_features, out_features),
      d_bias(out_features, 0.0),
      input_cache(1, 1),
      has_input_cache(false)
{
    assert(in_features > 0);
    assert(out_features > 0);
    assert(weight_init_scale > 0.0);

    std::mt19937 rng(random_seed);
    std::uniform_real_distribution<double> dist(-weight_init_scale, weight_init_scale);

    for (double& value : weight.data)
    {
        value = dist(rng);
    }
}

Matrix Linear::forward(const Matrix& input)
{
    assert(input.cols == in_features);
    input_cache = input;
    has_input_cache = true;
    return input.matmul(weight).add_row_vector(bias);
}

Matrix Linear::backward(const Matrix& grad_output)
{
    assert(has_input_cache);
    assert(grad_output.rows == input_cache.rows);
    assert(grad_output.cols == out_features);

    Matrix grad_weight = input_cache.transpose().matmul(grad_output);
    for (int i = 0; i < static_cast<int>(d_weight.data.size()); i++)
    {
        d_weight.data[i] += grad_weight.data[i];
    }

    for (int c = 0; c < out_features; c++)
    {
        double sum = 0.0;
        for (int r = 0; r < grad_output.rows; r++)
        {
            sum += grad_output(r, c);
        }
        d_bias[c] += sum;
    }

    return grad_output.matmul(weight.transpose());
}

void Linear::step(double learning_rate)
{
    assert(learning_rate > 0.0);

    for (int i = 0; i < static_cast<int>(weight.data.size()); i++)
    {
        weight.data[i] -= learning_rate * d_weight.data[i];
    }

    for (int i = 0; i < static_cast<int>(bias.size()); i++)
    {
        bias[i] -= learning_rate * d_bias[i];
    }

    zero_grad();
}

void Linear::zero_grad()
{
    for (double& value : d_weight.data)
    {
        value = 0.0;
    }

    for (double& value : d_bias)
    {
        value = 0.0;
    }
}
