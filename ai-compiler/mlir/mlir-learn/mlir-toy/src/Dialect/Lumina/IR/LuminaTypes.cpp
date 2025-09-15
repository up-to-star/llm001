#include "Dialect/Lumina/IR/LuminaTypes.h"

#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include <set>

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

Type LMTensorType::parse(::mlir::AsmParser &parser) {
    if (parser.parseLess()) {
        return Type();
    }

    llvm::SmallVector<int64_t, 4> dims;
    if (parser.parseDimensionList(dims, /*allowDynamic=*/true,
                                  /*withTrailingX=*/true)) {
        return Type();
    }
    auto typeLoc = parser.getCurrentLocation();
    Type elementType;
    if (parser.parseType(elementType)) {
        return Type();
    }
    if (parser.parseComma()) {
        return Type();
    }
    int64_t device_id = 0;
    if (parser.parseInteger(device_id)) {
        if (parser.parseGreater()) {
            return Type();
        }
    }
    return parser.getChecked<LMTensorType>(parser.getContext(), dims,
                                           elementType, device_id);
}

void LMTensorType::print(::mlir::AsmPrinter &printer) const {
    printer << "<";
    for (int64_t dim : getShape()) {
        if (dim < 0) {
            printer << "?" << 'x';
        } else {
            printer << dim << 'x';
        }
    }
    printer.printType(getElementType());
    printer << ", " << getDeviceId() << ">";
}

llvm::LogicalResult LMBufferType::verify(
    ::llvm::function_ref< ::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> devices) {
        if (std::set(devices.begin(), devices.end()).size() != devices.size()) {
            return emitError() << "devices must be unique";
        }
        for (auto id : devices) {
            if (id < 0) {
                return emitError() << "device_id must be non-negative";
            }
        }
        return ::llvm::success();
    }
}  // namespace mlir::lumina