#include "include\tokenizer.h"
Tokenizer::Tokenizer()
{
    char_to_id.fill(-1);
    for (int i = 0; i < wordList.length(); i++)
    {
        char c = wordList[i];
        char_to_id[c] = i;
        id_to_char.push_back(c);
    }
}
std::vector<int> Tokenizer::encode(std::string s)
{
    std::vector<int> id(s.size(), 0);
    for (int i = 0; i < s.size(); i++)
    {
        id[i] = char_to_id[s[i]];
        assert(id[i] != -1);
    }
    return id;
}
std::string Tokenizer::decode(std::vector<int> ids)
{
    std::string text(ids.size(), '\0');

    for (int i = 0; i < ids.size(); i++)
    {
        text[i] = id_to_char[ids[i]];
    }
    return text;
}
int Tokenizer::vocab_size()
{
    return wordList.size();
}