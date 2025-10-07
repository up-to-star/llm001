add_mlir_dialect_library(MLIRLuminaTransforms
  MarkDistributeParallelParameters.cpp
  ApplyDistributeTransform.cpp

  DEPENDS
  MLIRLuminaPassesIncGen

  LINK_LIBS PUBLIC
  MLIRLuminaDialect
)