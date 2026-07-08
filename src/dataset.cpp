#include "dataset.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "tokenizer.h"

namespace
{
std::string read_text_file(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open dataset file: " + path);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}
}

Dataset::Dataset(int block_size, const std::string& path)
    : block_size(block_size)
{
    assert(block_size > 0);

    Tokenizer tokenizer;
    std::string text = read_text_file(path);

    std::string filtered;
    filtered.reserve(text.size());
    for (char c : text)
    {
        if (tokenizer.can_encode(c))
        {
            filtered.push_back(c);
        }
    }

    data = tokenizer.encode(filtered);
    assert(static_cast<int>(data.size()) > block_size);
}

Dataset::Dataset(const std::vector<int>& token_ids, int block_size)
    : data(token_ids), block_size(block_size)
{
    assert(block_size > 0);
    assert(static_cast<int>(data.size()) > block_size);
}

int Dataset::size() const
{
    return static_cast<int>(data.size()) - block_size;
}

int Dataset::get_block_size() const
{
    return block_size;
}

int Dataset::token_at(int index) const
{
    assert(index >= 0);
    assert(index < static_cast<int>(data.size()));
    return data[index];
}

std::vector<int> Dataset::get_sample(int index) const
{
    assert(index >= 0);
    assert(index < size());

    std::vector<int> sample;
    sample.reserve(block_size + 1);

    for (int i = 0; i <= block_size; i++)
    {
        sample.push_back(data[index + i]);
    }

    return sample;
}
