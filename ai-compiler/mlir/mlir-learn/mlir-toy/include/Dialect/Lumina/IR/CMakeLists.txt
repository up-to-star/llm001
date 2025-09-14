set(LLVM_TARGET_DEFINITIONS LuminaTypes.td)

mlir_tablegen(LuminaDialect.h.inc --gen-dialect-decls -dialect=lumina)
mlir_tablegen(LuminaDialect.cpp.inc --gen-dialect-defs -dialect=lumina)

mlir_tablegen(LuminaTypes.h.inc -gen-typedef-decls -dialect=lumina)
mlir_tablegen(LuminaTypes.cpp.inc -gen-typedef-defs -dialect=lumina)

add_public_tablegen_target(LuminaDialectIncGen)