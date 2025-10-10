#include <memory>
#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/Transforms/Passes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "Utils/Key.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "apply-distribute-transform"

namespace mlir::lumina {
#define GEN_PASS_DEF_APPLYDISTRIBUTETRANSFORMPASS
#include "Dialect/Lumina/Transforms/Passes.h.inc"
}  // namespace mlir::lumina

using namespace ::mlir::lumina;
using namespace ::mlir;

struct ApplyDistributeTransformPass
    : mlir::lumina::impl::ApplyDistributeTransformPassBase<
          ApplyDistributeTransformPass> {
    using mlir::lumina::impl::ApplyDistributeTransformPassBase<
        ApplyDistributeTransformPass>::ApplyDistributeTransformPassBase;

    void runOnOperation() override;
};

void ApplyDistributeTransformPass::runOnOperation() {
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in: {0}\n", getPassName()));
    auto func = getOperation();
    LLVM_DEBUG(llvm::dbgs()
               << llvm::formatv("root op: {0}\n", func->getName()));

    auto dp_attr = llvm::dyn_cast_or_null<mlir::DistributeParallelAttr>(
        func->getAttr(KDPAttrName));
    if (!dp_attr) {
        llvm_unreachable("func must have dp_attr");
    }

    func->walk([&](mlir::Operation* op) {
        if (auto dis_op = llvm::dyn_cast_or_null<DistributeParallelOp>(op)) {
            if (dis_op.applyDistributeParallelism(dp_attr).succeeded()) {
                LLVM_DEBUG(llvm::dbgs()
                           << llvm::formatv("Apply DataParallelism to {0}\n",
                                            dis_op->getName()));
                op->erase();
            }
        }
    });

    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
}

std::unique_ptr<mlir::Pass> mlir::lumina::createApplyDistributeTransformPass() {
    return std::make_unique<ApplyDistributeTransformPass>();
}
