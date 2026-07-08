#pragma once

#include <cassert>
#include <string>
#include <vector>

class Dataset
{
public:
    Dataset(int block_size, const std::string& path = "data/input.txt");
    Dataset(const std::vector<int>& token_ids, int block_size);

    int size() const;
    int get_block_size() const;
    int token_at(int index) const;

    std::vector<int> get_sample(int index) const;

private:
    std::vector<int> data;
    int block_size;
};
