#include <memory>
#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/Transforms/Passes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "Utils/Key.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

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
    llvm::outs() << "run in: " << getPassName() << "\n";
    auto func = getOperation();
    llvm::outs() << "root op: " << func->getName() << "\n";

    auto dp_attr = llvm::dyn_cast_or_null<mlir::DistributeParallelAttr>(
        func->getAttr(KDPAttrName));
    if (!dp_attr) {
        llvm_unreachable("func must have dp_attr");
    }

    func->walk([&](Operation* op) {
        if (auto dis_op = llvm::dyn_cast_or_null<DistributeParallelOp>(op)) {
            if (dis_op.applyDistributeParallelism(dp_attr).succeeded()) {
                llvm::outs() << "Apply data parallelism to "
                             << dis_op->getName() << "\n";
                op->erase();
            }
        }
    });

    llvm::outs() << "run out: " << getPassName() << "\n";
}

std::unique_ptr<mlir::Pass> mlir::lumina::createApplyDistributeTransformPass() {
    return std::make_unique<ApplyDistributeTransformPass>();
}
