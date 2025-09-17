#include "Dialect/Lumina/IR/LuminaOps.h"

#include <algorithm>
#include <cstdint>

#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/IR/LuminaTypes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#define GET_OP_CLASSES
#include "Dialect/Lumina/IR/LuminaOps.cpp.inc"

namespace mlir::lumina {
void LuminaDialect::registerOps() {
    llvm::outs() << "register " << getDialectNamespace() << " op\n";
    addOperations<
#define GET_OP_LIST
#include "Dialect/Lumina/IR/LuminaOps.cpp.inc"
        >();
}

::llvm::LogicalResult LuminaGetTensorOp::verify() {
    auto device_id = getDeviceId();
    auto buffer = getBuffer();
    if (isa<BlockArgument>(buffer)) {
        auto buffer_type = cast<LMBufferType>(buffer.getType());
        auto device_ids = buffer_type.getDevices();
        for (auto id : device_ids) {
            if (id == device_id) {
                return ::llvm::success();
            }
        }
        return ::llvm::failure();
    }

    auto buffer_op = llvm::cast_or_null<LuminaBufferOp>(buffer.getDefiningOp());
    if (!buffer_op) {
        return ::llvm::failure();
    }
    for (auto tensor : buffer_op.getTensors()) {
        auto tensor_type = cast<LMTensorType>(tensor.getType());
        if (!tensor_type) {
            return ::llvm::failure();
        }
        if (tensor_type.getDeviceId() == device_id) {
            if (tensor_type != getType()) {
                return ::llvm::failure();
            }
            return ::llvm::success();
        }
    }
    return ::llvm::failure();
}

::llvm::LogicalResult LuminaBufferOp::verify() {
    auto tensors = getTensors();
    auto devices = cast<LMBufferType>(getType()).getDevices();
    if (tensors.size() == 0) {
        return ::llvm::failure();
    }

    for (auto [index, device_id, tensor] : llvm::enumerate(devices, tensors)) {
        auto tensor_type = cast<LMTensorType>(tensor.getType());
        if (device_id != tensor_type.getDeviceId()) {
            return ::llvm::failure();
        }
    }
    return ::llvm::success();
}

::llvm::LogicalResult LuminaSoftmaxOp::verify() {
    auto axis = getAxis();
    if (axis < 0) {
        return ::llvm::failure();
    }
    auto input_type = cast<LMTensorType>(getInput().getType());
    if (axis >= input_type.getShape().size()) {
        return ::llvm::failure();
    }

    return llvm::success();
}

::llvm::LogicalResult LuminaBufferCastOp::verify() {
    if (getNumResults() > 1) {
        if (!llvm::all_of(getResultTypes(),
                          [](Type type) { return isa<LMTensorType>(type); })) {
            return llvm::failure();
        }
    }

    if (getNumOperands() > 1) {
        if (!llvm::all_of(getOperandTypes(),
                          [](Type type) { return isa<LMTensorType>(type); })) {
            return llvm::failure();
        }
    }
    return llvm::success();
}

bool LuminaSoftmaxOp::supportDataParallelism() { return getAxis() != 0; }

LMTensorType LMTensorType::clone() const {
    return LMTensorType::get(getContext(), getShape(), getElementType(),
                             getDeviceId());
}

LMTensorType LMTensorType::clone(::mlir::Type elementType) const {
    return LMTensorType::get(getContext(), getShape(), elementType,
                             getDeviceId());
}

LMTensorType LMTensorType::clone(::mlir::ArrayRef<int64_t> shape) const {
    return LMTensorType::get(getContext(), shape, getElementType(),
                             getDeviceId());
}

LMTensorType LMTensorType::clone(::mlir::ArrayRef<int64_t> shape,
                                 int64_t device_id) const {
    return LMTensorType::get(getContext(), shape, getElementType(), device_id);
}

LMTensorType LMTensorType::clone(::mlir::ArrayRef<int64_t> shape,
                                 ::mlir::Type elementType) const {
    return LMTensorType::get(getContext(), shape, elementType, getDeviceId());
}

llvm::SmallVector<Type> splitTensor(const LMTensorType &tensor, int dim,
                                    ArrayRef<int64_t> device_ids) {
    llvm::SmallVector<Type> types;
    if (tensor.getRank() <= dim) {
        llvm::errs() << "out of dimension ranges";
        return {};
    }

    auto shapes = tensor.getShape();
    auto nums = device_ids.size();
    auto split_dim = shapes[dim];
    for (auto device_id : device_ids) {
        llvm::SmallVector<int64_t> new_shape(shapes.begin(), shapes.end());
        if (split_dim != ShapedType::kDynamic) {
            auto dim_value = split_dim / nums;
            new_shape[dim] = dim_value;
            split_dim -= dim_value;
            nums--;
        }
        auto new_tensor = tensor.clone(new_shape, device_id);
        types.push_back(new_tensor);
    }
    return types;
}

::llvm::LogicalResult LuminaSoftmaxOp::applyDataParallelism(
    ::mlir::DistributeParallelAttr attr) {
    auto dp_attr = llvm::dyn_cast_or_null<::mlir::DataParallelAttr>(attr);
    if (!dp_attr) {
        return ::llvm::failure();
    }

    if (!supportDataParallelism()) {
        return ::llvm::failure();
    }

    auto op = getOperation();
    // auto dp_num = dp_attr.getDPNums();
    auto device_ids = dp_attr.getDevices();
    OpBuilder builder(getOperation());
    builder.setInsertionPointAfter(getOperation());
    auto operands = getOperation()->getOperands();
    auto results = getOperation()->getResults();

    llvm::SmallVector<Operation *> ops;
    llvm::for_each(device_ids, [&](int64_t) {
        ops.push_back(builder.clone(*getOperation()));
    });
    for (auto [index, operand] : llvm::enumerate(operands)) {
        auto type = llvm::dyn_cast_or_null<LMTensorType>(operand.getType());
        auto types = splitTensor(type, 0, device_ids);
        auto cast = builder.create<lumina::LuminaBufferCastOp>(
            getLoc(), TypeRange(types), operands, attr);
        cast->moveAfter(op);
        for (auto [op_index, sub_op] : llvm::enumerate(ops)) {
            sub_op->setOperand(index, cast->getResult(op_index));
        }
    }

    for (auto [index, res] : llvm::enumerate(results)) {
        auto type = llvm::dyn_cast_or_null<LMTensorType>(res.getType());
        auto types = splitTensor(type, 0, device_ids);
        for (auto [op_index, sub_op] : llvm::enumerate(ops)) {
            sub_op->getResult(index).setType(types[op_index]);
        }
        llvm::SmallVector<Value> oprands;
        for (auto sub_op : ops) {
            oprands.push_back(sub_op->getResult(index));
        }
        auto cast = builder.create<lumina::LuminaBufferCastOp>(
            getLoc(), TypeRange{type}, oprands, attr);
        for (auto &use : res.getUses()) {
            use.set(cast->getOpResult(0));
        }
    }
    return ::llvm::success();
}

}  // namespace mlir::lumina