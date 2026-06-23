cmake_minimum_required(VERSION 3.10)
project(fa_camodel_test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN installation path
set(CANN_ROOT "/home/guohaojie/Ascend/cann-9.0.0/x86_64-linux" CACHE PATH "CANN installation path")
set(FA_OP_API_PATH "/home/guohaojie/Guo/FA算子/flash_attention_score/op_api" CACHE PATH "FlashAttention op_api path")

# Include directories
include_directories(
    ${CANN_ROOT}/include
    ${FA_OP_API_PATH}
)

# Library directories
link_directories(
    ${CANN_ROOT}/devlib/linux/x86_64
)

# Add executable with stub implementation
add_executable(fa_camodel_test
    main.cpp
    aclnn_fa_stub.cpp
)

# Link libraries
target_link_libraries(fa_camodel_test
    ascendcl
    nnopbase
    graph
    stdc++
    -Wl,--allow-shlib-undefined
    -Wl,--unresolved-symbols=ignore-in-shared-libs
)

# Set output directory
set_target_properties(fa_camodel_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Print configuration info
message(STATUS "CANN_ROOT: ${CANN_ROOT}")
message(STATUS "FA_OP_API_PATH: ${FA_OP_API_PATH}")
message(STATUS "Include directories: ${CANN_ROOT}/include, ${FA_OP_API_PATH}")
message(STATUS "Library directory: ${CANN_ROOT}/devlib/linux/x86_64")
