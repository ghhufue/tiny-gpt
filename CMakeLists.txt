cmake_minimum_required(VERSION 3.16)

project(tiny_gpt_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tiny_gpt_core
    src/tokenizer.cpp
    src/dataset.cpp
    src/dataloader.cpp
    src/embedding.cpp
)

target_include_directories(tiny_gpt_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_tokenizer
    test/test_tokenizer.cpp
)

add_executable(test_dataset
    test/test_dataset.cpp
)

add_executable(test_dataloader
    test/test_dataloader.cpp
)

add_executable(test_embedding
    test/test_embedding.cpp
)

target_link_libraries(test_tokenizer PRIVATE tiny_gpt_core)
target_link_libraries(test_dataset PRIVATE tiny_gpt_core)
target_link_libraries(test_dataloader PRIVATE tiny_gpt_core)
target_link_libraries(test_embedding PRIVATE tiny_gpt_core)
