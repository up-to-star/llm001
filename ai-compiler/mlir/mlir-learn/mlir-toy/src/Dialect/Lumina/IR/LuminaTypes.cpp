#include "Dialect/Lumina/IR/LuminaTypes.h"

#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Lumina/IR/LuminaTypes.cpp.inc"

namespace mlir::lumina {
void LuminaDialect::registerType() {
    llvm::outs() << "register " << getDialectNamespace() << " Types\n";
    addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Lumina/IR/LuminaTypes.cpp.inc"
        >();
}

::llvm::LogicalResult LMTensorType::verify(
    ::llvm::function_ref< ::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> shape, Type elementType, int64_t device_id) {
    if (device_id < 0) {
        return emitError() << "device_id must be non-negative";
    }
    if (!elementType.isIntOrFloat()) {
        return emitError() << "elementType must be int or float";
    }
    return ::llvm::success();
}
}  // namespace mlir::lumina