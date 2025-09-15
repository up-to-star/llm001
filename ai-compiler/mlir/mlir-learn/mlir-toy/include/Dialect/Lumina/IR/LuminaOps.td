include "Dialect/Lumina/IR/LuminaAttrs.td"
include "Dialect/Lumina/IR/LuminaConstraints.td"

class LuminaOp<string mnemonic, list<Trait> traits = []>
    : Op<LuminaDialect, mnemonic, traits> {
    let summary = "Lumina operation";

    let description = "$_name op";
}

class LuminaUnaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)>
    : LuminaOp<mnemonic, traits#[]> {
    let arguments = !con((ins
        OperandType: $input),
        attributes
        );

    let results = (outs
        resultType: $result
    );
}

class LuminaBinaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)>
    : LuminaOp<mnemonic, traits#[]> {
    let arguments = !con((ins
        OperandType: $lhs,
        OperandType: $rhs),
        attributes
        );

    let results = (outs
        resultType: $result
    );
}

def LuminaConstOp : LuminaOp<"const", []> {
    let arguments = (ins
        ElementsAttr: $value
    );

    let results = (outs
        AnyLMTensor: $result
    );
}


def LuminaSoftmaxOp : LuminaUnaryOp<"softmax", AnyLMTensor, AnyLMTensor, [], (ins I64Attr: $axis)> {
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "::mlir::Value": $input, "int64_t": $axis), [{
            $_state.addOperands(input);
            $_state.getOrAddProperties<Properties>().axis = $_builder.getIntegerAttr(odsBuilder.getIntegerType(64, true), axis);
            $_state.addTypes(input.getType());
            }]>
    ];
}


def LuminaExpOp : LuminaUnaryOp<"exp", AnyLMTensor, AnyLMTensor, [], (ins)> {

    let builders = [
        OpBuilder<(ins "::mlir::Value": $input), [{
            $_state.addOperands(input);
            $_state.addTypes(input.getType());
            }]>
    ];
}

def LuminaAddOp : LuminaBinaryOp<"add", AnyLMTensor, AnyLMTensor, [], (ins)>;
def LuminaSubOp : LuminaBinaryOp<"sub", AnyLMTensor, AnyLMTensor, [], (ins)>;
def LuminaMulOp : LuminaBinaryOp<"mul", AnyLMTensor, AnyLMTensor, [], (ins)>;
def LuminaDivOp : LuminaBinaryOp<"div", AnyLMTensor, AnyLMTensor, [], (ins)>;

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