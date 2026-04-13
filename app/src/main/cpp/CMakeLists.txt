cmake_minimum_required(VERSION 3.22.1)

project(soctest_ncnn)

set(NCNN_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/ncnn/ncnn-20260113-android-vulkan/${ANDROID_ABI})
set(NCNN_LIB_DIR ${NCNN_ROOT}/lib)
set(ANDROID_OMP_LIB ${ANDROID_NDK}/toolchains/llvm/prebuilt/${ANDROID_HOST_TAG}/lib/clang/${CMAKE_CXX_COMPILER_VERSION}/lib/linux/aarch64/libomp.a)

add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${NCNN_LIB_DIR}/libncnn.a)

add_library(omp STATIC IMPORTED)
set_target_properties(omp PROPERTIES IMPORTED_LOCATION ${ANDROID_OMP_LIB})

foreach(lib_name
    glslang
    SPIRV
    MachineIndependent
    GenericCodeGen
    OSDependent
    glslang-default-resource-limits
)
    add_library(${lib_name} STATIC IMPORTED)
    set_target_properties(${lib_name} PROPERTIES IMPORTED_LOCATION ${NCNN_LIB_DIR}/lib${lib_name}.a)
endforeach()

add_library(soctest_ncnn SHARED ncnn_bridge.cpp)

target_include_directories(soctest_ncnn PRIVATE ${NCNN_ROOT}/include)

target_link_libraries(
    soctest_ncnn
    ncnn
    glslang
    SPIRV
    MachineIndependent
    GenericCodeGen
    OSDependent
    glslang-default-resource-limits
    omp
    vulkan
    log
    android
    z
)
