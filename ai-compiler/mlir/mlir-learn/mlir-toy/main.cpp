#include <filesystem>
#include <memory>

#include "Dialect/Lumina/IR/LuminaAttrs.h"
#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/IR/LuminaEnums.h"
#include "Dialect/Lumina/IR/LuminaOps.h"
#include "Dialect/Lumina/IR/LuminaTypes.h"
#include "Dialect/Lumina/Transforms/Passes.h"
#include "Dialect/Lumina/Transforms/Passes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "Utils/File.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "Utils/Key.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

void testDialect() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    auto dialect = context.getOrLoadDialect<mlir::lumina::LuminaDialect>();
    dialect->sayHello();
}

void typeBrief() {
    // 文件定义：llvm-project/mlir/include/mlir/IR/BuiltinTypes.td
    auto context = new mlir::MLIRContext;

    // 浮点数，每种位宽和标准定义一个
    auto f32 = mlir::Float32Type::get(context);
    llvm::outs() << "F32类型 :\t";
    f32.dump();

    auto bf16 = mlir::BFloat16Type::get(context);
    llvm::outs() << "BF16类型 :\t";
    bf16.dump();

    // Index 类型，机器相关的整数类型
    auto index = mlir::IndexType::get(context);
    llvm::outs() << "Index 类型 :\t";
    index.dump();

    // 整数类型, 参数: 位宽&&有无符号
    auto i32 = mlir::IntegerType::get(context, 32);
    llvm::outs() << "I32 类型 :\t";
    i32.dump();
    auto ui16 =
        mlir::IntegerType::get(context, 16, mlir::IntegerType::Unsigned);
    llvm::outs() << "UI16 类型 :\t";
    ui16.dump();

    // 张量类型,表示的是数据，不会有内存的布局信息。
    auto static_tensor = mlir::RankedTensorType::get({1, 2, 3}, f32);
    llvm::outs() << "静态F32 张量类型 :\t";
    static_tensor.dump();
    // 动态张量
    auto dynamic_tensor =
        mlir::RankedTensorType::get({mlir::ShapedType::kDynamic, 2, 3}, f32);
    llvm::outs() << "动态F32 张量类型 :\t";
    dynamic_tensor.dump();

    // Memref类型：表示内存
    auto basic_memref = mlir::MemRefType::get({1, 2, 3}, f32);
    llvm::outs() << "静态F32 内存类型 :\t";
    basic_memref.dump();
    // 带有布局信息的内存

    auto stride_layout_memref = mlir::MemRefType::get(
        {1, 2, 3}, f32, mlir::StridedLayoutAttr::get(context, 1, {6, 3, 1}));
    llvm::outs() << "连续附带布局信息的 F32 内存类型 :\t";
    stride_layout_memref.dump();
    // 使用affine 表示布局信息的内存
    auto affine_memref = mlir::MemRefType::get(
        {1, 2, 3}, f32,
        mlir::StridedLayoutAttr::get(context, 1, {6, 3, 1}).getAffineMap());
    llvm::outs() << "连续附带 affine 布局信息的 F32 内存类型 :\t";
    affine_memref.dump();
    // 动态连续附带 affine 布局信息的内存
    auto dynamic_affine_memref = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, 2, 3}, f32,
        mlir::StridedLayoutAttr::get(context, 1,
                                     {mlir::ShapedType::kDynamic, 3, 1})
            .getAffineMap());
    llvm::outs() << "连续附带 affine 布局信息的动态 F32 内存类型 :\t";
    dynamic_affine_memref.dump();
    // 具有内存层级信息的内存
    auto L1_memref = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, 2, 3}, f32,
        mlir::StridedLayoutAttr::get(context, 1,
                                     {mlir::ShapedType::kDynamic, 3, 1})
            .getAffineMap(),
        1);
    llvm::outs() << "处于L1层级的 F32 内存类型 :\t";
    L1_memref.dump();
    // gpu 私有内存层级的内存
    context->getOrLoadDialect<mlir::gpu::GPUDialect>();
    auto gpu_memref = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, 2, 3}, f32,
        mlir::StridedLayoutAttr::get(context, 1,
                                     {mlir::ShapedType::kDynamic, 3, 1})
            .getAffineMap(),
        mlir::gpu::AddressSpaceAttr::get(context,
                                         mlir::gpu::AddressSpace::Private));
    llvm::outs()
        << "连续附带 affine 布局信息的动态 F32 Gpu Private内存类型 :\t";
    gpu_memref.dump();

    // 向量类型,定长的一段内存
    auto vector_type = mlir::VectorType::get(3, f32);
    llvm::outs() << "F32 1D向量类型 :\t";
    vector_type.dump();

    auto vector_2D_type = mlir::VectorType::get({3, 3}, f32);
    llvm::outs() << "F32 2D向量类型 :\t";
    vector_2D_type.dump();
    delete context;
}

void myType() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    // 加载注册方言
    auto dialect = context.getOrLoadDialect<mlir::lumina::LuminaDialect>();
    auto tm_tensor = mlir::lumina::LMTensorType::get(
        &context, {1, 2, 3}, mlir::Float32Type::get(&context), 1);
    tm_tensor.dump();

    auto dy_tm_tensor = mlir::lumina::LMTensorType::get(
        &context, {mlir::ShapedType::kDynamic, 2, 3},
        mlir::Float32Type::get(&context), 1);
    dy_tm_tensor.dump();
}

void attrBrief() {
    auto context = std::make_shared<mlir::MLIRContext>();
    context->getOrLoadDialect<mlir::lumina::LuminaDialect>();

    auto f32_attr =
        mlir::FloatAttr::get(mlir::Float32Type::get(context.get()), 1.0);
    llvm::outs() << "F32 属性 :\t";
    f32_attr.dump();

    auto i32_attr =
        mlir::IntegerAttr::get(mlir::IntegerType::get(context.get(), 32), 10);
    llvm::outs() << "I32 属性 :\t";
    i32_attr.dump();

    auto stride_layout_attr =
        mlir::StridedLayoutAttr::get(context.get(), 1, {6, 3, 1});
    llvm::outs() << "StridedLayout 属性 :\t";
    stride_layout_attr.dump();

    auto str_attr = mlir::StringAttr::get(context.get(), "hello");
    llvm::outs() << "String 属性 :\t";
    str_attr.dump();

    auto str_ref_attr = mlir::SymbolRefAttr::get(str_attr);
    llvm::outs() << "SymbolRef 属性 :\t";
    str_ref_attr.dump();

    auto type_attr = mlir::TypeAttr::get(mlir::lumina::LMTensorType::get(
        context.get(), {1, 2, 3}, mlir::Float32Type::get(context.get()), 1));
    llvm::outs() << "Type 属性 :\t";
    type_attr.dump();

    auto unit_attr = mlir::UnitAttr::get(context.get());
    llvm::outs() << "Unit Attribute: \t";
    unit_attr.dump();

    auto i64_arr_attr = mlir::DenseI64ArrayAttr::get(context.get(), {1, 2, 3});
    llvm::outs() << "Dense I64 Array Attribute: \t";
    i64_arr_attr.dump();

    auto dense_attr = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get({2, 2},
                                    mlir::Float32Type::get(context.get())),
        llvm::ArrayRef<float>{1, 2, 3, 4});
    llvm::outs() << "Dense Elements Attribute: \t";
    dense_attr.dump();
}

void testAttr() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::lumina::LuminaDialect>();
    auto nchw = mlir::lumina::Layout::NCHW;
    llvm::outs() << "NCHW Layout Attribute: "
                 << mlir::lumina::stringifyEnum(nchw) << "\n";

    auto nhwc = mlir::lumina::Layout::NHWC;
    llvm::outs() << "NHWC Layout Attribute: "
                 << mlir::lumina::stringifyEnum(nhwc) << "\n";

    auto dp_attr = mlir::lumina::DataParallelismAttr::get(&context, 2);
    llvm::outs() << "DataParallelism Attribute: \t";
    dp_attr.dump();
}

void testOp() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::lumina::LuminaDialect>();

    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    auto module = builder.create<mlir::ModuleOp>(loc, "Lumina");
    builder.setInsertionPointToStart(module.getBody());

    // ConstOp
    auto f32 = mlir::Float32Type::get(&context);
    auto shape = mlir::SmallVector<int64_t>({2, 2});
    auto const_value_1 =
        mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)1));
    auto const_value_2 =
        mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)2));
    auto tensor_type_1 =
        mlir::lumina::LMTensorType::get(&context, shape, f32, 0);
    auto tensor_type_2 =
        mlir::lumina::LMTensorType::get(&context, shape, f32, 1);
    auto const_1 = builder.create<mlir::lumina::LuminaConstOp>(
        loc, tensor_type_1,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_1));
    auto const_2 = builder.create<mlir::lumina::LuminaConstOp>(
        loc, tensor_type_1,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_1));

    auto const_3 = builder.create<mlir::lumina::LuminaConstOp>(
        loc, tensor_type_2,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_2));
    auto const_4 = builder.create<mlir::lumina::LuminaConstOp>(
        loc, tensor_type_2,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_2));

    llvm::outs() << "Const tensor in device 0: \n";
    const_1->dump();
    llvm::outs() << "Const tensor in device 1: \n";
    const_3->dump();

    // BufferOp
    auto buffer_op = builder.create<mlir::lumina::LuminaBufferOp>(
        loc, mlir::ValueRange{const_1, const_3});
    llvm::outs() << "Buffer Op: \n";
    buffer_op->dump();

    // GetTensorOp
    auto get_tensor_op_1 = builder.create<mlir::lumina::LuminaGetTensorOp>(
        loc, tensor_type_1, buffer_op, 0);
    auto get_tensor_op_2 = builder.create<mlir::lumina::LuminaGetTensorOp>(
        loc, tensor_type_2, buffer_op, 1);
    llvm::outs() << "GetTensor Op: \n";
    get_tensor_op_1->dump();
    get_tensor_op_2->dump();

    // Softmax op
    auto softmax_op =
        builder.create<mlir::lumina::LuminaSoftmaxOp>(loc, get_tensor_op_1, 1);
    llvm::outs() << "Softmax Op: \n";
    softmax_op->dump();

    // Exp Op
    auto exp_op =
        builder.create<mlir::lumina::LuminaExpOp>(loc, get_tensor_op_2);
    llvm::outs() << "Exp Op: \n";
    exp_op->dump();

    // all to all Op
    auto out_buffer_op = builder.create<mlir::lumina::LuminaBufferOp>(
        loc, mlir::ValueRange{const_2, const_4});
    auto all_to_all_op = builder.create<mlir::lumina::LuminaAllToAllOp>(
        loc, buffer_op, out_buffer_op);
    llvm::outs() << "AllToAll Op: \n";
    all_to_all_op->dump();
}

void testInterface() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::lumina::LuminaDialect>();

    auto f32 = mlir::Float32Type::get(&context);
    auto dim = mlir::ShapedType::kDynamic;
    auto shape = mlir::SmallVector<int64_t>{dim, dim, 64};
    auto tensor_type = mlir::lumina::LMTensorType::get(&context, shape, f32, 0);
    tensor_type.dump();

    auto shaped_type = mlir::dyn_cast_or_null<mlir::ShapedType>(tensor_type);
    if (shaped_type) {
        llvm::outs() << "Shaped Type: \n";
        shaped_type.dump();
    }
    auto clone_type = shaped_type.clone(f32);
    llvm::outs() << "Clone Type: \n";
    clone_type.dump();

    auto dp_attr = mlir::lumina::DataParallelismAttr::get(&context, 2);
    llvm::outs()
        << dp_attr.getAbstractAttribute().getName()
        << " has DistributeParallelAttr: "
        << dp_attr
               .hasPromiseOrImplementsInterface<mlir::DistributeParallelAttr>()
        << "\n";

    llvm::outs() << dp_attr.getAbstractAttribute().getName()
                 << " has DataParallelAttr: "
                 << dp_attr.getAbstractAttribute().hasInterface(
                        mlir::DataParallelAttr::getInterfaceID())
                 << "\n";
}

void testIRStruct() {
    const char* ir =
        R"(func.func @insertion_point_outside_loop(%t : tensor<?xf32>, %sz : index,
                                        %idx : index) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %blank = tensor.empty() : tensor<5xf32>

  %r = scf.for %iv = %c0 to %sz step %c5 iter_args(%bb = %t) -> (tensor<?xf32>) {
    %iv_i32 = arith.index_cast %iv : index to i32
    %f = arith.sitofp %iv_i32 : i32 to f32

    %filled = linalg.fill ins(%f : f32) outs(%blank : tensor<5xf32>) -> tensor<5xf32>

    %inserted = tensor.insert_slice %filled into %bb[%idx][5][1] : tensor<5xf32> into tensor<?xf32>
    scf.yield %inserted : tensor<?xf32>
  }
  return %r : tensor<?xf32>
})";
    auto context = mlir::MLIRContext();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::affine::AffineDialect>();
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.getOrLoadDialect<mlir::tensor::TensorDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module;

    if (mlir::utils::file::ParseStr(context, module, ir).failed()) {
        llvm::outs() << "Parse IR string failed\n";
    }

    auto file = std::filesystem::current_path() / "ir_struct.mlir";
    if (mlir::utils::file::PrintToFile(module.get(), file.string().c_str())
            .failed()) {
        llvm::outs() << "Print to file failed\n";
    }
}

mlir::ModuleOp getModule(mlir::OpBuilder& builder) {
    auto loc = builder.getUnknownLoc();
    auto context = builder.getContext();
    auto module = builder.create<mlir::ModuleOp>(loc, "Lumina");
    builder.setInsertionPointToStart(module.getBody());
    auto f32 = mlir::Float32Type::get(context);
    auto dy_dim = 128;
    auto dy_shape = mlir::SmallVector<int64_t>({dy_dim, dy_dim, 24});
    auto dy_tensor_type =
        mlir::lumina::LMTensorType::get(context, dy_shape, f32, 0);
    auto func_type =
        mlir::FunctionType::get(context, {dy_tensor_type}, {dy_tensor_type});
    auto func =
        builder.create<mlir::func::FuncOp>(loc, KEntryPointName, func_type);
    func->setAttr(KHostFunc, builder.getUnitAttr());
    func->setAttr(KDPAttrName,
                  mlir::lumina::DataParallelismAttr::get(context, 2));

    auto block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    // Softmax Op
    mlir::Value softmax_op = builder.create<mlir::lumina::LuminaSoftmaxOp>(
        loc, block->getArgument(0), 1);
    softmax_op =
        builder.create<mlir::lumina::LuminaSoftmaxOp>(loc, softmax_op, 1);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{softmax_op});
    return module;
}

void testPass() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);

    context.getOrLoadDialect<mlir::lumina::LuminaDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = getModule(builder);
    mlir::PassManager pm(&context);
    mlir::lumina::MarkDistributeParallelParameterOptions
        mark_distribute_parallel_option{.DPNums = 3, .TPNums = 1};
    pm.addPass(mlir::lumina::createMarkDistributeParallelParameter(
        mark_distribute_parallel_option));
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::lumina::createApplyDistributeTransformPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::lumina::createDeviceRegionFusionPass());
    module->dump();
    llvm::outs() << "=========\n";
    if (pm.run(module).failed()) {
        llvm::outs() << "Pass run failed\n";
    }
    llvm::outs() << "after pass: \n";
    module->dump();
}

int main() {
    // testDialect();
    // typeBrief();
    // myType();
    // attrBrief();
    // testAttr();
    // testOp();
    // testInterface();
    // testIRStruct();
    testPass();
    return 0;
}