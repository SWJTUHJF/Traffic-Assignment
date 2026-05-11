cmake_minimum_required(VERSION 3.16)

project(TrafficAssignmentCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(traffic_assignment
    src/main.cpp
    src/network.cpp
    src/parser.cpp
    src/shortest_path.cpp
    src/solver_base.cpp
    src/link_based.cpp
)

target_include_directories(traffic_assignment PRIVATE include)
