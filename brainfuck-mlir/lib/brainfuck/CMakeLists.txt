
file(GLOB BF_SRC *.cc)


add_mlir_dialect_library(
    MLIRBrainfuck

    ${BF_SRC}


    DEPENDS
    MLIRBrainfuckOpsIncGen
    MLIRBrainfuckTypeInterfacesIncGen

    LINK_LIBS PUBLIC 
	MLIRIR
    MLIRInferTypeOpInterface
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTransforms
)

mlir_check_link_libraries(MLIRBrainfuck)

target_include_directories(MLIRBrainfuck PRIVATE ${CMAKE_BINARY_DIR}/brainfuck-mlir/include)
target_include_directories(MLIRBrainfuck PRIVATE ${PROJECT_SOURCE_DIR}/brainfuck-mlir/include)