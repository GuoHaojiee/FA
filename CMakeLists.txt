cmake_minimum_required(VERSION 3.16)

# Compiler must be set before project() to take effect
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(CMAKE_C_COMPILER "$ENV{ASCEND_HOME_PATH}/bin/bisheng")
    set(CMAKE_CXX_COMPILER "$ENV{ASCEND_HOME_PATH}/bin/bisheng")
endif()

project(fa_camodel_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH not set. Please run: source set_env.sh")
endif()
set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})

if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend950PR_9599")
endif()

if(NOT DEFINED RUN_MODE)
    set(RUN_MODE "sim")
endif()

include_directories(
    ${ASCEND_PATH}/include
    ${ASCEND_PATH}/x86_64-linux/include
)

link_directories(
    ${ASCEND_PATH}/x86_64-linux/lib64
    ${ASCEND_PATH}/tools/simulator/${SOC_VERSION}/lib
)

add_executable(test_fa_builtin
    ${CMAKE_SOURCE_DIR}/op_test_template/test_fa_builtin.cpp)

target_compile_options(test_fa_builtin PRIVATE
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

target_link_libraries(test_fa_builtin PRIVATE
    $<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>
    $<$<STREQUAL:${RUN_MODE},npu>:runtime>
    ascendcl
    nnopbase
    opapi_transformer
    opapi_oam
    stdc++
    m
    dl
    pthread
    c_sec
)

set_target_properties(test_fa_builtin PROPERTIES
    BUILD_RPATH "${ASCEND_PATH}/x86_64-linux/lib64:${ASCEND_PATH}/tools/simulator/${SOC_VERSION}/lib"
    INSTALL_RPATH "${ASCEND_PATH}/x86_64-linux/lib64:${ASCEND_PATH}/tools/simulator/${SOC_VERSION}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/op_test_template"
)
