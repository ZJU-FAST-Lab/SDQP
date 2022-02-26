cmake_minimum_required(VERSION 2.8.3)

project(sdqp)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")

find_package(Eigen3 REQUIRED)

find_package(osqp REQUIRED)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_example example/sdqp_example.cpp)

target_link_libraries(${PROJECT_NAME}_example  osqp::osqp)