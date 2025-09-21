set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LuminaOpt)
add_public_tablegen_target(MLIRLuminaPassesIncGen)
add_dependencies(mlir-headers MLIRLuminaPassesIncGen) 