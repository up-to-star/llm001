#pragma once

#include "Dialect/Lumina/IR/LuminaEnums.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "Interfaces/DistributeParallelismInterfaces.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/Lumina/IR/LuminaAttrs.h.inc"
