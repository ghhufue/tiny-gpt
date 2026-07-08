#include "dataloader.h"

DataLoader::DataLoader(const Dataset& dataset)
    : dataset(dataset), rng(std::random_device{}())
{
}

Batch DataLoader::get_batch(int batch_size)
{
    assert(batch_size > 0);

    Batch batch;
    batch.batch_size = batch_size;
    batch.block_size = dataset.get_block_size();
    batch.tokens.reserve(batch_size * (batch.block_size + 1));

    std::uniform_int_distribution<int> dist(0, dataset.size() - 1);

    for (int b = 0; b < batch_size; b++)
    {
        int index = dist(rng);

        for (int i = 0; i <= batch.block_size; i++)
        {
            batch.tokens.push_back(dataset.token_at(index + i));
        }
    }

    return batch;
}
