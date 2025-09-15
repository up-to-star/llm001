include "mlir/IR/EnumAttr.td"
include "Dialect/Lumina/IR/LuminaTypes.td"

def Lumina_LAYOUT_NCHW : I32EnumAttrCase<"NCHW", 0>;
def Lumina_LAYOUT_NHWC : I32EnumAttrCase<"NHWC", 1>;

def Lumina_Layout : I32EnumAttr<"Layout", "Layout of tensor", [Lumina_LAYOUT_NCHW, Lumina_LAYOUT_NHWC]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::lumina";
}

def LLH_LayoutAttr : EnumAttr<LuminaDialect, Lumina_Layout, "Layout"> {
    let assemblyFormat = "`<` $value `>`";
    
    let extraClassDeclaration = [{
        bool isChannelLast();
    }];
}