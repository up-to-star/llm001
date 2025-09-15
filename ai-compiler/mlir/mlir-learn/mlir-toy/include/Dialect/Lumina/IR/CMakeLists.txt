set(LLVM_TARGET_DEFINITIONS LuminaAttrs.td)

mlir_tablegen(LuminaDialect.h.inc --gen-dialect-decls -dialect=lumina)
mlir_tablegen(LuminaDialect.cpp.inc --gen-dialect-defs -dialect=lumina)

mlir_tablegen(LuminaTypes.h.inc -gen-typedef-decls -dialect=lumina)
mlir_tablegen(LuminaTypes.cpp.inc -gen-typedef-defs -dialect=lumina)

mlir_tablegen(LuminaEnums.h.inc -gen-enum-decls -dialect=lumina)
mlir_tablegen(LuminaEnums.cpp.inc -gen-enum-defs -dialect=lumina)

mlir_tablegen(LuminaAttrs.h.inc -gen-attrdef-decls -dialect=lumina)
mlir_tablegen(LuminaAttrs.cpp.inc -gen-attrdef-defs -dialect=lumina)

add_public_tablegen_target(LuminaDialectIncGen)