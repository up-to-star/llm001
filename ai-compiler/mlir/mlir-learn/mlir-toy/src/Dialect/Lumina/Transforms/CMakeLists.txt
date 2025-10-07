add_mlir_dialect_library(MLIRLuminaTransforms
  MarkDistributeParallelParameters.cpp
  ApplyDistributeTransform.cpp
  DeviceRegionFusion.cpp

  DEPENDS
  MLIRLuminaPassesIncGen

  LINK_LIBS PUBLIC
  MLIRLuminaDialect
  MLIRTransforms
)