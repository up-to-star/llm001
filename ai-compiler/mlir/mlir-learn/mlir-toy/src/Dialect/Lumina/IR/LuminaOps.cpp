#include "Dialect/Lumina/IR/LuminaOps.h"

#include <algorithm>
#include <cstdint>

#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/IR/LuminaTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Block.h"
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

}  // namespace mlir::lumina