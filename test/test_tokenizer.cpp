#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "../include/tokenizer.h"

int main() {
    Tokenizer tokenizer;

    // 1. 测试 vocab size
    assert(tokenizer.vocab_size() == 96);

    // 2. 测试单个字符 encode
    {
        std::vector<int> ids = tokenizer.encode("abc");
        assert(ids.size() == 3);
        assert(ids[0] == 0);  // a
        assert(ids[1] == 1);  // b
        assert(ids[2] == 2);  // c
    }

    // 3. 测试大小写字母
    {
        std::vector<int> ids = tokenizer.encode("zAZ");
        assert(ids.size() == 3);
        assert(ids[0] == 25); // z
        assert(ids[1] == 26); // A
        assert(ids[2] == 51); // Z
    }

    // 4. 测试数字
    {
        std::vector<int> ids = tokenizer.encode("0123456789");
        assert(ids.size() == 10);
        assert(ids[0] == 52);
        assert(ids[9] == 61);
    }

    // 5. 测试空格和标点
    {
        std::string text = "hello, world!";
        std::vector<int> ids = tokenizer.encode(text);
        std::string decoded = tokenizer.decode(ids);
        assert(decoded == text);
    }

    // 6. 测试常用符号
    {
        std::string text = "a+b=c*d/(e-f)";
        std::vector<int> ids = tokenizer.encode(text);
        std::string decoded = tokenizer.decode(ids);
        assert(decoded == text);
    }

    // 7. 测试括号和代码符号
    {
        std::string text = "int main() { return 0; }";
        std::vector<int> ids = tokenizer.encode(text);
        std::string decoded = tokenizer.decode(ids);
        assert(decoded == text);
    }

    // 8. 测试换行和制表符
    {
        std::string text = "hello\n\tworld";
        std::vector<int> ids = tokenizer.encode(text);
        std::string decoded = tokenizer.decode(ids);
        assert(decoded == text);
    }

    // 9. 测试完整 vocab round-trip
    {
        std::string vocab =
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,!?;:'\"-_()[]{}<>/\\+=*@#$%^&|~\n\t";

        std::vector<int> ids = tokenizer.encode(vocab);
        std::string decoded = tokenizer.decode(ids);

        assert(decoded == vocab);
        assert(ids.size() == tokenizer.vocab_size());
    }

    std::cout << "All tokenizer tests passed!" << std::endl;

    return 0;
}