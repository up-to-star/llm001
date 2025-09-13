#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

int main() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    auto dialect = context.getOrLoadDialect<mlir::lumina::LuminaDialect>();
    dialect->sayHello();
    return 0;
}