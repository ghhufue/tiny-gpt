#pragma once

#include <cassert>
#include <random>
#include <vector>

#include "dataset.h"

struct Batch
{
    std::vector<int> tokens; // shape: [batch_size, block_size + 1]

    int batch_size;
    int block_size;

    int input(int b, int t) const
    {
        assert(b >= 0);
        assert(b < batch_size);
        assert(t >= 0);
        assert(t < block_size);
        return tokens[b * (block_size + 1) + t];
    }

    int target(int b, int t) const
    {
        assert(b >= 0);
        assert(b < batch_size);
        assert(t >= 0);
        assert(t < block_size);
        return tokens[b * (block_size + 1) + t + 1];
    }
};

class DataLoader
{
public:
    explicit DataLoader(const Dataset& dataset);

    Batch get_batch(int batch_size);

private:
    const Dataset& dataset;
    std::mt19937 rng;
};
