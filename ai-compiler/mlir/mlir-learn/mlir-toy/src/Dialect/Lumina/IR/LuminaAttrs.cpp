#include "Dialect/Lumina/IR/LuminaAttrs.h"

#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/IR/LuminaEnums.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/Lumina/IR/LuminaAttrs.cpp.inc"
#include "Dialect/Lumina/IR/LuminaEnums.cpp.inc"

namespace mlir::lumina {
void LuminaDialect::registerAttrs() {
    llvm::outs() << "register " << getDialectNamespace() << " Attr\n";
    addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/Lumina/IR/LuminaAttrs.cpp.inc"
        >();
}

bool LayoutAttr::isChannelLast() { return getValue() == Layout::NHWC; }


}  // namespace mlir::lumina
