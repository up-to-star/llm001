include "mlir/IR/EnumAttr.td"
include "Dialect/Lumina/IR/LuminaEnums.td"
include "Interfaces/DistributeParallelismInterfaces.td"

class LuminaAttr<string name, string attrMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute">
    : AttrDef<LuminaDialect, name, traits, baseCppClass> {
    let mnemonic = attrMnemonic;

    let attrName = dialect.name # "." # attrMnemonic;

    let genStorageClass = 1;

    let hasStorageCustomConstructor = 0;

    let skipDefaultBuilders = 0;

    let genVerifyDecl = 0;
}

def Lumina_DataParallelism : LuminaAttr<"DataParallelism", "DP", [DataParallelAttr]> {
   let parameters = (ins
    "int64_t": $DP_nums,
    ArrayRefParameter<"int64_t">: $devices
   );

   let builders = [
    AttrBuilder<(ins "int64_t": $DP_nums),
        [{
            llvm::SmallVector<int64_t> device_ids;
            for (auto i : llvm::index_range(0, DP_nums)) {
                device_ids.push_back(i);
            }
            return $_get($_ctxt, DP_nums, device_ids);
        }]>
   ];

   let assemblyFormat = [{
   `<`
    `DP` `=` $DP_nums `,`
    `devices` `=` $devices
   `>`
   }];
}