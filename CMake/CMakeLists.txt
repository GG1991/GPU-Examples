
cmake_minimum_required(VERSION 3.11)

project(cmake-test LANGUAGES CXX)

option(ENABLE_CUDA "Enable CUDA" OFF)
if(ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUDA")
endif()

if(ENABLE_CUDA)
  file(GLOB SOURCES main.cpp func.cu func.cpp)
  set_source_files_properties(func.cpp PROPERTIES LANGUAGE CUDA)
else()
  file(GLOB SOURCES main.cpp func.cpp)
endif()

add_executable(cmake-test ${SOURCES}) 
set_property(TARGET cmake-test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

