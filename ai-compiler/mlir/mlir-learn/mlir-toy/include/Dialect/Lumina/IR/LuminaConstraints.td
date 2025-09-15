include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

def AnyLMTensor : Type<And<[CPred<"::mlir::isa<::mlir::lumina::LMTensorType>($_self)">, 
                            CPred<"::mlir::cast<::mlir::lumina::LMTensorType>($_self).getShape().size() > 0">]>>;

def AnyLMBuffer : Type<And<[CPred<"::mlir::isa<::mlir::lumina::LMBufferType>($_self)">, 
                            CPred<"::mlir::cast<::mlir::lumina::LMBufferType>($_self).getDevices().size() > 0">]>>;