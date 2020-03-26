cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

foreach(path ${CMAKE_PREFIX_PATH})
  message("Path: " ${path})
endforeach(path)

#set(KALDI_DIR /mnt/cephfs2/asr/users/fanlu/kaldi2)
set(KALDI_DIR /root/kaldi)
set(TARGET emo2)
#set(TARGET inference)
set(SRC emo2.cc)
#set(SRC inference.cpp)

include_directories(${KALDI_DIR}/src ${KALDI_DIR}/tools/openfst/include)
link_directories(${KALDI_DIR}/src/lib)

add_definitions(-O3 -std=c++11)

find_package(Torch REQUIRED)

add_executable(${TARGET} ${SRC})
message("aaa:${_GLIBCXX_USE_CXX11_ABI}")
target_link_libraries(${TARGET} "${TORCH_LIBRARIES}" kaldi-util kaldi-base kaldi-matrix kaldi-feat)
#target_link_libraries(${TARGET} "${TORCH_LIBRARIES}")
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 11)
