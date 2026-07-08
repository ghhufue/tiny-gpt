#include "embedding.h"

#include <cassert>
#include <cmath>

Embedding::Embedding(int vocab_size, int d_model, int max_positions,
                     double weight_init_scale, unsigned int random_seed)
    : vocab_size(vocab_size), d_model(d_model), max_positions(max_positions)
{
    assert(vocab_size > 0);
    assert(d_model > 0);
    assert(max_positions > 0);
    assert(weight_init_scale > 0.0);

    token_table.resize(vocab_size * d_model);
    position_table.resize(max_positions * d_model);

    std::mt19937 rng(random_seed);
    std::uniform_real_distribution<double> dist(-weight_init_scale, weight_init_scale);

    for (double& value : token_table)
    {
        value = dist(rng);
    }

    for (int pos = 0; pos < max_positions; pos++)
    {
        for (int dim = 0; dim < d_model; dim++)
        {
            double exponent = static_cast<double>(2 * (dim / 2)) / d_model;
            double angle = static_cast<double>(pos) / std::pow(10000.0, exponent);
            position_table[pos * d_model + dim] = (dim % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
}

int Embedding::get_vocab_size() const
{
    return vocab_size;
}

int Embedding::get_d_model() const
{
    return d_model;
}

int Embedding::get_max_positions() const
{
    return max_positions;
}

std::vector<double> Embedding::token_embedding(int token_id) const
{
    assert(token_id >= 0);
    assert(token_id < vocab_size);

    std::vector<double> result(d_model);
    for (int dim = 0; dim < d_model; dim++)
    {
        result[dim] = token_value(token_id, dim);
    }
    return result;
}

std::vector<double> Embedding::position_encoding(int position) const
{
    assert(position >= 0);
    assert(position < max_positions);

    std::vector<double> result(d_model);
    for (int dim = 0; dim < d_model; dim++)
    {
        result[dim] = position_value(position, dim);
    }
    return result;
}

std::vector<double> Embedding::embed(int token_id, int position) const
{
    assert(token_id >= 0);
    assert(token_id < vocab_size);
    assert(position >= 0);
    assert(position < max_positions);

    std::vector<double> result(d_model);
    for (int dim = 0; dim < d_model; dim++)
    {
        result[dim] = token_value(token_id, dim) + position_value(position, dim);
    }
    return result;
}

std::vector<double> Embedding::forward(const std::vector<int>& ids) const
{
    assert(static_cast<int>(ids.size()) <= max_positions);

    std::vector<double> output(ids.size() * d_model);
    for (int pos = 0; pos < static_cast<int>(ids.size()); pos++)
    {
        assert(ids[pos] >= 0);
        assert(ids[pos] < vocab_size);

        for (int dim = 0; dim < d_model; dim++)
        {
            output[pos * d_model + dim] = token_value(ids[pos], dim) + position_value(pos, dim);
        }
    }
    return output;
}

double Embedding::token_value(int token_id, int dim) const
{
    return token_table[token_id * d_model + dim];
}

double Embedding::position_value(int position, int dim) const
{
    return position_table[position * d_model + dim];
}
