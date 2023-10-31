add_mlir_dialect(BrainfuckOps brainfuck)
# add_mlir_interface(BrainfuckTypeInterfaces)





set(LLVM_TARGET_DEFINITIONS BrainfuckTypeInterfaces.td)
mlir_tablegen(BrainfuckTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(BrainfuckTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRBrainfuckTypeInterfacesIncGen)


