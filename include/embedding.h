#pragma once

#include <random>
#include <vector>

class Embedding
{
public:
    Embedding(int vocab_size, int d_model, int max_positions,
              double weight_init_scale = 0.01, unsigned int random_seed = 42);

    int get_vocab_size() const;
    int get_d_model() const;
    int get_max_positions() const;

    std::vector<double> token_embedding(int token_id) const;
    std::vector<double> position_encoding(int position) const;
    std::vector<double> embed(int token_id, int position) const;

    // Returns flattened shape: [ids.size(), d_model].
    std::vector<double> forward(const std::vector<int>& ids) const;

private:
    int vocab_size;
    int d_model;
    int max_positions;

    std::vector<double> token_table;    // shape: [vocab_size, d_model]
    std::vector<double> position_table; // shape: [max_positions, d_model]

    double token_value(int token_id, int dim) const;
    double position_value(int position, int dim) const;
};
