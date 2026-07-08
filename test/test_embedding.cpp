#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "../include/embedding.h"

namespace
{
bool near(double a, double b)
{
    return std::abs(a - b) < 1e-9;
}
}

int main()
{
    Embedding embedding(10, 6, 8, 0.01, 123);

    assert(embedding.get_vocab_size() == 10);
    assert(embedding.get_d_model() == 6);
    assert(embedding.get_max_positions() == 8);

    {
        std::vector<double> pos0 = embedding.position_encoding(0);
        assert(pos0.size() == 6);
        assert(near(pos0[0], 0.0));
        assert(near(pos0[1], 1.0));
        assert(near(pos0[2], 0.0));
        assert(near(pos0[3], 1.0));
        assert(near(pos0[4], 0.0));
        assert(near(pos0[5], 1.0));
    }

    {
        std::vector<double> token = embedding.token_embedding(3);
        std::vector<double> pos = embedding.position_encoding(2);
        std::vector<double> combined = embedding.embed(3, 2);

        assert(token.size() == 6);
        assert(pos.size() == 6);
        assert(combined.size() == 6);

        for (int dim = 0; dim < 6; dim++)
        {
            assert(near(combined[dim], token[dim] + pos[dim]));
        }
    }

    {
        std::vector<int> ids = {1, 2, 3};
        std::vector<double> output = embedding.forward(ids);
        assert(output.size() == 18);

        std::vector<double> expected = embedding.embed(2, 1);
        for (int dim = 0; dim < 6; dim++)
        {
            assert(near(output[1 * 6 + dim], expected[dim]));
        }
    }

    std::cout << "All embedding tests passed!" << std::endl;

    return 0;
}
