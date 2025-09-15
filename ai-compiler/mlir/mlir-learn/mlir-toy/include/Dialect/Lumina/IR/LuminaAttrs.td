include "mlir/IR/EnumAttr.td"
include "Dialect/Lumina/IR/LuminaEnums.td"

class LuminaAttr<string name, string attrMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute">
    : AttrDef<LuminaDialect, name, traits, baseCppClass> {
    let mnemonic = attrMnemonic;

    let attrName = dialect.name # "." # attrMnemonic;

    let genStorageClass = 1;

    let hasStorageCustomConstructor = 0;

    let skipDefaultBuilders = 0;

    let genVerifyDecl = 0;
}

def Lumina_DataParallelism : LuminaAttr<"DataParallelism", "DP", []> {
   let parameters = (ins
    "int64_t": $DP_nums
   );

   let assemblyFormat = [{
   `<`
   `DP` `=` $DP_nums
   `>`
   }];
}