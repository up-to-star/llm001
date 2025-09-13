add_mlir_dialect_library(MLIRLuminaDialect
    LuminaDialect.cpp

    DEPENDS
    LuminaDialectIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorDialect
)