
cmake_minimum_required(VERSION 3.11)

project(cmake-test LANGUAGES CXX)

option(ENABLE_CUDA "Enable CUDA" OFF)
if(ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA")
endif()

include_directories(include)

if(ENABLE_CUDA)
	file(GLOB SOURCES main.cpp func.cu)
else()
	file(GLOB SOURCES main.cpp func.cpp)
endif()

add_executable(cmake-test ${SOURCES}) 
