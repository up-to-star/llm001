include "Dialect/Lumina/IR/LuminaAttrs.td"
include "Dialect/Lumina/IR/LuminaConstraints.td"
include "Interfaces/FusionRegionInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

class LuminaOp<string mnemonic, list<Trait> traits = []>
    : Op<LuminaDialect, mnemonic, traits> {
    let summary = "Lumina operation";

    let description = "$_name op";
}

class LuminaUnaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)>
    : LuminaOp<mnemonic, [DeclareOpInterfaceMethods<DistributeParallelOp>] # traits> {
    let arguments = !con((ins
        OperandType: $input),
        attributes
        );

    let results = (outs
        resultType: $result
    );
}

class LuminaBinaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)>
    : LuminaOp<mnemonic, [DeclareOpInterfaceMethods<DistributeParallelOp>] # traits> {
    let arguments = !con((ins
        OperandType: $lhs,
        OperandType: $rhs),
        attributes
        );

    let results = (outs
        resultType: $result
    );
}

class LuminaElewiseUnaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)>
    : LuminaUnaryOp<mnemonic, OperandType, resultType, [DeclareOpInterfaceMethods<SupportDataParallelismOp>] # traits, attributes>;

class LuminaElewiseBinaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)>
    : LuminaBinaryOp<mnemonic, OperandType, resultType, [DeclareOpInterfaceMethods<SupportDataParallelismOp>] # traits, attributes>;

def LuminaConstOp : LuminaOp<"const", []> {
    let arguments = (ins
        ElementsAttr: $value
    );

    let results = (outs
        AnyLMTensor: $result
    );
}


def LuminaSoftmaxOp : LuminaUnaryOp<"softmax", AnyLMTensor, AnyLMTensor, [DeclareOpInterfaceMethods<SupportDataParallelismOp, ["applyDataParallelism", "supportDataParallelism"]>], (ins I64Attr: $axis)> {
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "::mlir::Value": $input, "int64_t": $axis), [{
            $_state.addOperands(input);
            $_state.getOrAddProperties<Properties>().axis = $_builder.getI64IntegerAttr(axis);
            $_state.addTypes(input.getType());
            }]>
    ];
}


def LuminaExpOp : LuminaElewiseUnaryOp<"exp", AnyLMTensor, AnyLMTensor, [], (ins)> {

    let builders = [
        OpBuilder<(ins "::mlir::Value": $input), [{
            $_state.addOperands(input);
            $_state.addTypes(input.getType());
            }]>
    ];
}

def LuminaAddOp : LuminaElewiseBinaryOp<"add", AnyLMTensor, AnyLMTensor, [], (ins)>;
def LuminaSubOp : LuminaElewiseBinaryOp<"sub", AnyLMTensor, AnyLMTensor, [], (ins)>;
def LuminaMulOp : LuminaElewiseBinaryOp<"mul", AnyLMTensor, AnyLMTensor, [], (ins)>;
def LuminaDivOp : LuminaElewiseBinaryOp<"div", AnyLMTensor, AnyLMTensor, [], (ins)>;

def LuminaAllToAllOp : LuminaOp<"all_to_all", []> {
    let arguments = (ins
        AnyLMBuffer: $input,
        AnyLMBuffer: $output
    );
}

def LuminaBufferOp : LuminaOp<"buffer", []> {
    let arguments = (ins
        Variadic<AnyLMTensor>: $tensors
    );

    let results = (outs
        AnyLMBuffer: $result
    );

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "::mlir::ValueRange": $tensors), [{
            $_state.addOperands(tensors);
            ::llvm::SmallVector<int64_t> devices;
            for (auto tensor : tensors) {
                devices.push_back(::mlir::cast<::mlir::lumina::LMTensorType>(tensor.getType()).getDeviceId());
            }
            $_state.addTypes(::mlir::lumina::LMBufferType::get($_state.getContext(), devices));
            }]>
    ];
}

def LuminaGetTensorOp : LuminaOp<"get_tensor", []> {
    let description = "从 buffer 中获取指定 device_id 的 tensor";

    let arguments = (ins
        AnyLMBuffer: $buffer,
        I64Attr: $device_id
    );

    let results = (outs
        AnyLMTensor: $result
    );

    let hasVerifier = 1;
}

def LuminaPrintOp : LuminaOp<"print", []> {
    let arguments = (ins
        AnyType: $input
    );
}

def LuminaBufferCastOp : LuminaOp<"buffer_cast", []> {
    let description = "对数据切分的标记";

    let arguments = (ins
        Variadic<LMTensorOrBuffer>: $inputs,
        DistributeParallelAttr: $distribute_attr
    );

    let results = (outs
        Variadic<LMTensorOrBuffer>: $outputs
    );

    let hasVerifier = 1;
}

def LuminaDeviceKernelOp: LuminaOp<"device_region",[
        DeclareOpInterfaceMethods<FusionRegionOpInterface,["FusionOps"]>,AffineScope, AutomaticAllocationScope,IsolatedFromAbove, OpAsmOpInterface]>{
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        I64Attr:$device_id,
        Variadic<AnyType>:$args
    );

    let regions = (region SizedRegion<1>:$region);

    let results = (outs
        Variadic<AnyType>:$result);

    let hasVerifier = 1;
}