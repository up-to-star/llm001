include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "Dialect/Lumina/IR/LuminaDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class LuminaType<string name, string typeMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : TypeDef<LuminaDialect, name, traits, baseCppClass> {
    let mnemonic = typeMnemonic;
    let typeName = dialect.name # "." # typeMnemonic;
}

def LuminaTensorType : LuminaType<"LMTensor", "lm_tensor", []> {
    let summary = "Lumina tensor type";

    let description = "Lumina tensor type";

    let parameters = (ins
        ArrayRefParameter<"int64_t">: $shape,
        "Type": $elementType,
        "int64_t": $device_id
    );

    let genStorageClass = 1;

    let hasStorageCustomConstructor = 0;

    let builders = [
        TypeBuilder<(ins 
        "::mlir::ArrayRef<int64_t>": $shape, 
        "::mlir::Type": $elementType), [{
            return $_get(elementType.getContext(), shape, elementType, 0);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
    // let assemblyFormat = "`<`$shape `,`$elementType`,`$device_id`>`";

    let skipDefaultBuilders = 0;

    let genVerifyDecl = 1;
}

