#include <cassert>
#include <iostream>
#include <vector>

#include "../include/dataset.h"

int main()
{
    std::vector<int> tokens = {0, 1, 2, 3, 4, 5, 6};
    Dataset dataset(tokens, 3);

    assert(dataset.size() == 4);

    {
        std::vector<int> sample = dataset.get_sample(0);
        assert((sample == std::vector<int>{0, 1, 2, 3}));
    }

    {
        std::vector<int> sample = dataset.get_sample(3);
        assert((sample == std::vector<int>{3, 4, 5, 6}));
    }

    {
        Dataset file_dataset(8);
        assert(file_dataset.size() > 0);
        assert(file_dataset.get_sample(0).size() == 9);
    }

    std::cout << "All dataset tests passed!" << std::endl;

    return 0;
}
