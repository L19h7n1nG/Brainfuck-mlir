add_subdirectory(include)
add_subdirectory(lib)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)





add_llvm_executable(braincc braincc.cc)

llvm_update_compile_flags(braincc)


target_include_directories(braincc PRIVATE ${CMAKE_BINARY_DIR}/brainfuck-mlir/include)
target_include_directories(braincc PRIVATE ${PROJECT_SOURCE_DIR}/brainfuck-mlir/include)

target_link_libraries(
    braincc
    PRIVATE
    ${llvm_libs}
    ${dialect_libs}
    MLIRBrainfuck
    MLIROptLib
)

if(ipo_supported)
    message(STATUS "target braincc IPO / LTO enabled")
    set_property(TARGET braincc PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif()



