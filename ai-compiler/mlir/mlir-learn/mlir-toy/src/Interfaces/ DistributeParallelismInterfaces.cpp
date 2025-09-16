#include "Interfaces/DistributeParallelismInterfaces.h"
#include "Interfaces/DistributeParallelismOpInterfaces.cpp.inc"
#include "Interfaces/DistributeParallelismAttrInterfaces.cpp.inc"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
::llvm::LogicalResult DistributeParallelOp::applyDistributeParallelism(
    const ::mlir::DistributeParallelAttr &attr) {
    if (isa<DataParallelAttr>(attr)) {
        if (!isa<SupportDataParallelismOp>(getOperation())) {
            return llvm::failure();
        }
        return dyn_cast<SupportDataParallelismOp>(getOperation())
            .applyDataParallelism(attr);
    } else {
        llvm_unreachable("unsupported distribute parallelism attr");
    }
    return llvm::success();
}

bool DistributeParallelOp::supportDistributeParallelism() {
    if (isa<SupportDataParallelismOp>(getOperation())) {
        return dyn_cast<SupportDataParallelismOp>(getOperation())
            .supportDataParallelism();
    } else {
        llvm_unreachable("unsupported op type");
    }
    return false;
}

}  // namespace mlir
