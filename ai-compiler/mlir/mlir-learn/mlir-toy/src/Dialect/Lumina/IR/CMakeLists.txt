add_mlir_dialect_library(MLIRLuminaDialect
    LuminaDialect.cpp
    LuminaTypes.cpp

    DEPENDS
    LuminaDialectIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorDialect
)