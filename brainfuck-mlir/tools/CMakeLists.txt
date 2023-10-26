get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)


message("hello")

add_llvm_tool(brainfuck-lsp-server
    brainfuck-lsp-server.cc
)

target_include_directories(brainfuck-lsp-server PRIVATE ${CMAKE_BINARY_DIR}/brainfuck-mlir/include)
target_include_directories(brainfuck-lsp-server PRIVATE ${PROJECT_SOURCE_DIR}/brainfuck-mlir/include)

target_link_libraries(brainfuck-lsp-server
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRLspServerLib

  MLIRBrainfuck
)