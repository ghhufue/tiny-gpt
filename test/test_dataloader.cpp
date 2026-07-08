#include <cassert>
#include <iostream>
#include <vector>

#include "../include/dataloader.h"

int main()
{
    std::vector<int> tokens = {0, 1, 2, 3, 4, 5, 6};
    Dataset dataset(tokens, 3);
    DataLoader dataloader(dataset);

    Batch batch = dataloader.get_batch(4);
    assert(batch.batch_size == 4);
    assert(batch.block_size == 3);
    assert(batch.tokens.size() == 16);

    for (int b = 0; b < batch.batch_size; b++)
    {
        for (int t = 0; t < batch.block_size; t++)
        {
            assert(batch.target(b, t) == batch.input(b, t) + 1);
        }
    }

    std::cout << "All dataloader tests passed!" << std::endl;

    return 0;
}
