cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

foreach(path ${CMAKE_PREFIX_PATH})
  message("Path: " ${path})
endforeach(path)

set(KALDI_DIR ~/Documents/dev/workspace/kaldi)
set(TARGET emo2)
set(SRC emo2.cc)

include_directories(${KALDI_DIR}/src ${KALDI_DIR}/tools/openfst/include)
link_directories(${KALDI_DIR}/src/lib)

add_definitions(-O3 -g -std=c++11)

find_package(Torch REQUIRED)

add_executable(${TARGET} ${SRC})
message("aaa:${_GLIBCXX_USE_CXX11_ABI}")
target_link_libraries(${TARGET} "${TORCH_LIBRARIES}" kaldi-util kaldi-base kaldi-matrix kaldi-feat)
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 11)
