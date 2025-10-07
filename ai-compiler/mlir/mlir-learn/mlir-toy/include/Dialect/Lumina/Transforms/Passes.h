#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::lumina {
std::unique_ptr<mlir::Pass> createApplyDistributeTransformPass();

void populateBufferCastOpCanonicalizationPatterns(RewritePatternSet &patterns);

void populateDeviceRegionFusionPatterns(RewritePatternSet &patterns);

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Dialect/Lumina/Transforms/Passes.h.inc"
}  // namespace mlir::lumina
