include "mlir/IR/Interfaces.td"


def DistributeParallelAttr : AttrInterface<"DistributeParallelAttr"> {
    let description = "Distribute parallel attribute";

    let cppNamespace = "::mlir";

    let methods = [];

    let extraClassDeclaration = "";

    let extraSharedClassDeclaration = "";
}

def DataParallelAttr : AttrInterface<"DataParallelAttr", [DistributeParallelAttr]> {
    let description = "Data parallel attribute";

    let cppNamespace = "::mlir";

    let methods = [
        InterfaceMethod<[{
            DP 数量.
        }], 
        "int64_t", "getDPNums">,

        InterfaceMethod<[{
            设备编号.
        }],
        "::llvm::ArrayRef<int64_t>", "getDevices">,
    ];

    let extraClassDeclaration = "";

    let extraSharedClassDeclaration = "";
}

def SupportDataParallelismOp : OpInterface<"SupportDataParallelismOp"> {
    let description = "Support data parallelism op interface";

    let cppNamespace = "::mlir";

    let methods = [
        InterfaceMethod<
            /*desc=*/ "进行数据并行的变换",
            /*returnType=*/ "::llvm::LogicalResult",
            /*methodName=*/ "applyDataParallelism",
            /*args=*/ (ins "::mlir::DistributeParallelAttr":$attr),
            /*methodBody=*/ "",
            /*defaultImpl=*/ [{
                llvm::outs() << "a\n";
                return llvm::failure();
            }]
        >,

        InterfaceMethod<
            /*desc=*/ "进行数据并行的变换",
            /*returnType=*/ "bool",
            /*methodName=*/ "supportDataParallelism",
            /*args=*/ (ins),
            /*methodBody=*/ "",
            /*defaultImpl=*/ [{
                Operation *op = $_op.getOperation();
                if (op->getNumOperands() == 0) {
                    return true;
                }
                auto base_type = op->getOperand(0).getType();
                if (!isa<mlir::ShapedType>(base_type)) {
                    return false;
                }
                for (auto operand : op->getOperands()) {
                    if (operand.getType() != base_type) {
                        return false;
                    }
                }
                return true;
            }]
        >,
    ];

    let extraClassDeclaration = "";

    let extraSharedClassDeclaration = "";
}

def DistributeParallelOp : OpInterface<"DistributeParallelOp"> {
    let description = "Distribute parallel op interface";

    let cppNamespace = "::mlir";

    let methods = [];

    let extraClassDeclaration = "";

    let extraSharedClassDeclaration = [{
        ::llvm::LogicalResult applyDistributeParallelism(const ::mlir::DistributeParallelAttr &attr);

        bool supportDistributeParallelism();
    }];
}