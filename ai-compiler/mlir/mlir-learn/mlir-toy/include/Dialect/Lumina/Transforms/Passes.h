#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::lumina {
std::unique_ptr<mlir::Pass> createApplyDistributeTransformPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Dialect/Lumina/Transforms/Passes.h.inc"
}  // namespace mlir::lumina
