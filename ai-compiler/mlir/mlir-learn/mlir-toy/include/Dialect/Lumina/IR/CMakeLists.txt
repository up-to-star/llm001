set(LLVM_TARGET_DEFINITIONS LuminaDialect.td)

mlir_tablegen(LuminaDialect.h.inc --gen-dialect-decls -dialect=Lumina)
mlir_tablegen(LuminaDialect.cpp.inc --gen-dialect-defs -dialect=Lumina)

add_public_tablegen_target(LuminaDialectIncGen)