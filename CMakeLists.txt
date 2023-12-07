cmake_minimum_required(VERSION 3.0.2...3.21.0)
project(neural_network)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
  include
  src
)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable (${PROJECT_NAME} 
    ${SOURCES}
)
target_compile_options(${PROJECT_NAME} PRIVATE -g3)
target_link_libraries(${PROJECT_NAME})


file(GLOB_RECURSE TESTS "test/*.cpp")
set(TEST_NAME "test_gen")

add_executable (${TEST_NAME} 
    ${TESTS}
)
target_link_libraries(${TEST_NAME})
