add_mlir_dialect_library(MLIRLuminaDialect
    LuminaDialect.cpp
    LuminaTypes.cpp
    LuminaAttrs.cpp
    LuminaOps.cpp

    DEPENDS
    LuminaDialectIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorDialect
)