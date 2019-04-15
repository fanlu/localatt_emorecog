cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

# list(APPEND CMAKE_PREFIX_PATH "/Users/fanlu/workspace/workspace/mklml_mac_2019.0.3.20190220")
foreach(path ${CMAKE_PREFIX_PATH})
  message("Path: " ${path})
endforeach(path)
find_package(Torch REQUIRED)

add_executable(inference inference.cpp)
target_link_libraries(inference "${TORCH_LIBRARIES}")
set_property(TARGET inference PROPERTY CXX_STANDARD 11)
