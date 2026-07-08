#pragma once
#include <vector>
#include <string>
#include <array>
#include <cassert>
class Tokenizer
{
public:
    Tokenizer();
    std::vector<int> encode(const std::string& s) const;
    std::string decode(const std::vector<int>& token) const;
    int vocab_size() const;
    bool can_encode(char c) const;

private:
    const std::string wordList = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-_()[]{}<>/\\+=*@#$%^&|~\n\t";
    std::array<int, 128> char_to_id;
    std::vector<char> id_to_char;
};
