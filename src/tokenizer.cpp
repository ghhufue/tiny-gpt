#include "tokenizer.h"

Tokenizer::Tokenizer()
{
    char_to_id.fill(-1);
    for (int i = 0; i < wordList.length(); i++)
    {
        unsigned char c = static_cast<unsigned char>(wordList[i]);
        char_to_id[c] = i;
        id_to_char.push_back(wordList[i]);
    }
}

std::vector<int> Tokenizer::encode(const std::string& s) const
{
    std::vector<int> id(s.size(), 0);
    for (int i = 0; i < s.size(); i++)
    {
        assert(can_encode(s[i]));
        id[i] = char_to_id[static_cast<unsigned char>(s[i])];
        assert(id[i] != -1);
    }
    return id;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const
{
    std::string text(ids.size(), '\0');

    for (int i = 0; i < ids.size(); i++)
    {
        text[i] = id_to_char[ids[i]];
    }
    return text;
}

int Tokenizer::vocab_size() const
{
    return wordList.size();
}

bool Tokenizer::can_encode(char c) const
{
    unsigned char index = static_cast<unsigned char>(c);
    return index < char_to_id.size() && char_to_id[index] != -1;
}
