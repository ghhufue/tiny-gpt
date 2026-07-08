#pragma once
#include <vector>
#include <string>
#include <array>
#include <cassert>
class Tokenizer
{
public:
    Tokenizer();
    std::vector<int> encode(std::string s);
    std::string decode(std::vector<int> token);
    int vocab_size();

private:
    const std::string wordList = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-_()[]{}<>/\\+=*@#$%^&|~\n\t";
    std::array<int, 128> char_to_id;
    std::vector<char> id_to_char;
};