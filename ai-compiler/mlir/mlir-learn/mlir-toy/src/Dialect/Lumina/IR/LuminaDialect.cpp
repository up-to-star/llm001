#include "Dialect/Lumina/IR/LuminaDialect.h"

#include "Dialect/Lumina/IR/LuminaDialect.cpp.inc"
#include "llvm/Support/raw_ostream.h"

namespace mlir::lumina {
void LuminaDialect::initialize() {
    llvm::outs() << "initializeint " << getDialectNamespace() << "\n";
    registerType();
    registerAttrs();
    registerOps();
}

LuminaDialect::~LuminaDialect() {
    llvm::outs() << "destroy " << getDialectNamespace() << "\n";
}

void LuminaDialect::sayHello() {
    llvm::outs() << "hello " << getDialectNamespace() << "\n";
}
}  // namespace mlir::lumina