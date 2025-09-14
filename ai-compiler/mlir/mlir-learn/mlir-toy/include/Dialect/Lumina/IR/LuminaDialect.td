include "mlir/IR/DialectBase.td"

def LuminaDialect : Dialect {
    let name = "lumina";

    let summary = "Lumina dialect";

    let description = "Lumina dialect";

    let cppNamespace = "::mlir::lumina";

    let dependentDialects = ["::mlir::tensor::TensorDialect"];

    let discardableAttrs = (ins);

    let extraClassDeclaration = [{
        static void sayHello();
        void registerType();
    }];

    let hasConstantMaterializer = 0;

    let hasNonDefaultDestructor = 1;

    let hasOperationAttrVerify = 0;

    let hasRegionArgAttrVerify = 0;

    let hasRegionResultAttrVerify = 0;

    let hasOperationInterfaceFallback = 0;

    let useDefaultAttributePrinterParser = 0;

    let useDefaultTypePrinterParser = 1;

    let hasCanonicalizer = 0;

    let isExtensible = 0;

    let usePropertiesForAttributes = 1;
}