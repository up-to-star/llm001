#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/IR/LuminaAttrs.h"
#include "Dialect/Lumina/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Utils/Key.h"

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
    llvm::outs() << "run in: " << getPassName() << "\n";
    auto module = getOperation();
    llvm::outs() << "root op: " << module->getName() << "\n";
    llvm::outs() << "DP Nums: " << DPNums << "\n";
    llvm::outs() << "TP Nums: " << TPNums << "\n";
    llvm::outs() << "EP Nums: " << EPNums << "\n";

    if (TPNums != 1) {
        llvm::errs() << "TPNums not support now\n";
    }
    if (DPNums != 1) {
        auto dp_attr = DataParallelismAttr::get(&getContext(), DPNums);
        module->walk(
            [&dp_attr](func::FuncOp op) { op->setAttr(KDPAttrName, dp_attr); });
    }

    llvm::outs() << "run out: " << getPassName() << "\n";
}