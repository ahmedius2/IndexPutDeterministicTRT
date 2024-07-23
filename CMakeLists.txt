cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(index_put_deterministic_trt_plugin LANGUAGES CXX CUDA)
#set(TRT_ROOT "/home/a249s197/work/libraries/TensorRT-10.0.1.6")
set(TRT_ROOT "/home/a249s197/work/libraries/TensorRT-10.2.0.19")
set(TRT_INCLUDES "${TRT_ROOT}/include;${TRT_ROOT}/samples/common;${TRT_ROOT}/samples")
set(TRT_LIBS "${TRT_ROOT}/lib")


set(CMAKE_CUDA_ARCHITECTURES "61;87;")

add_library(index_put_lib SHARED
  src/nonZeroKernel.cu  src/nonZeroKernel.h  src/sampleNonZeroPlugin.cpp src/sampleNonZeroPlugin.h
  ${TRT_ROOT}/samples/common/logger.cpp 
)

# Request that index_put_lib be built with -std=c++14
# As this is a public compile feature anything that links to index_put_lib
# will also build with -std=c++14
target_compile_features(index_put_lib PUBLIC cxx_std_14)

# We need to explicitly state that we need all CUDA files in the index_put
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( index_put_lib
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(index_put_lib PUBLIC  ${TRT_INCLUDES} /usr/local/cuda/include)
target_link_libraries(index_put_lib PUBLIC 
  "${TRT_LIBS}/libnvinfer.so"
  "${TRT_LIBS}/libnvinfer_plugin.so"
  "${TRT_LIBS}/libnvonnxparser.so"
)

#include dirs:
# /home/a249s197/work/libraries/TensorRT/samples/common
# /home/a249s197/work/libraries/TensorRT-10.0.1.6/include

add_executable(index_put_test src/main.cpp)
set_target_properties(index_put_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(index_put_test PUBLIC  ${TRT_INCLUDES})
target_link_libraries(index_put_test PRIVATE index_put_lib)
