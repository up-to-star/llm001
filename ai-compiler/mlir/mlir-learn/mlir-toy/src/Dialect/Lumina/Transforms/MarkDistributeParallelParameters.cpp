#include <memory>

#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/IR/LuminaAttrs.h"
#include "Dialect/Lumina/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Utils/Key.h"

#define DEBUG_TYPE "mark-distribute-parallel-parameters"

namespace mlir::lumina {
#define GEN_PASS_DEF_MARKDISTRIBUTEPARALLELPARAMETER
#include "Dialect/Lumina/Transforms/Passes.h.inc"
}  // namespace mlir::lumina

using namespace ::mlir::lumina;
using namespace ::mlir;

struct MarkDistributeParallelParameter
    : mlir::lumina::impl::MarkDistributeParallelParameterBase<
          MarkDistributeParallelParameter> {
    using mlir::lumina::impl::MarkDistributeParallelParameterBase<
        MarkDistributeParallelParameter>::MarkDistributeParallelParameterBase;

    void runOnOperation() override;
};

void MarkDistributeParallelParameter::runOnOperation() {
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in: {0}\n", getPassName()));
    auto module = getOperation();
    LLVM_DEBUG(llvm::dbgs()
               << llvm::formatv("root op: {0}\n", module->getName()));
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("DP Nums: {0}\n", DPNums));
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("TP Nums: {0}\n", TPNums));
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("EP Nums: {0}\n", EPNums));

    if (TPNums != 1) {
        llvm::errs() << "TPNums not support now\n";
        signalPassFailure();
        return;
    }
    if (DPNums != 1) {
        auto dp_attr = DataParallelismAttr::get(&getContext(), DPNums);
        module->walk(
            [&dp_attr](func::FuncOp op) { op->setAttr(KDPAttrName, dp_attr); });
    }

    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
}