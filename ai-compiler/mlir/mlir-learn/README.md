## build llvm-project
```bash
cd llvm-project
mkdir build
cd build

cmake -G Ninja ../llvm \
  -DCMAKE_INSTALL_PREFIX=/home/cyj/studyspace/llm001/ai-compiler/mlir/mlir-learn/install \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

ninja install
```

## build mlir-toy
```bash
mkdir build
cd build
cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=/home/cyj/studyspace/llm001/ai-compiler/mlir/mlir-learn/install
```

## MLIR 内建Type
### 基本标量类型
- iN：整数类型，N 表示位宽（如 i1、i8、i32、i64）。i1 通常用于表示布尔值。
- fN：浮点类型，N 表示位宽（如 f16、f32、f64），遵循 IEEE 754 标准。
- index：索引类型，用于表示内存地址或循环索引，位宽与目标架构的指针宽度一致（如 32 位或 64 位）。等价与CPP中的`size_t`

```cpp
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
```
![alt text](./imgs/image.png)

### 张量类型
- 静态张量：维度大小固定，如 tensor<4x8xi32> 表示 2 维张量（4 行 8 列，元素类型为 i32）。
- 动态张量：维度大小可变，用 ? 表示，如 tensor<?x?xf32> 表示 2 维动态浮点张量。
- 零维张量：标量的张量包装，如 tensor<i32> 等价于标量 i32（但语义上属于张量范畴）。

```cpp
 // 张量类型,表示的是数据，不会有内存的布局信息。
auto static_tensor = mlir::RankedTensorType::get({1, 2, 3}, f32);
llvm::outs() << "静态F32 张量类型 :\t";
static_tensor.dump();
// 动态张量
auto dynamic_tensor =
    mlir::RankedTensorType::get({mlir::ShapedType::kDynamic, 2, 3}, f32);
llvm::outs() << "动态F32 张量类型 :\t";
dynamic_tensor.dump();
```
![alt text](./imgs/image.png)

### memref 类型
memref类型是用于描述内存中多维数组的核心类型，它不仅包含数组的逻辑形状，还明确指定了内存布局（如元素在内存中的排列方式）和地址空间，是连接高层抽象（如张量计算）与底层内存操作的关键桥梁。其核心作用包括：
- 描述多维数组的逻辑形状（如 2×3 的矩阵）
- 定义元素在内存中的物理布局（如步长、偏移量）
- 指定数据所在的地址空间（如全局内存、局部内存、设备内存）
- 支持底层代码生成（如 LLVM IR）时的内存访问优化。

`memref` 类型的完整语法为： `memref<ShapexElementType, Layout, AddressSpace>`
- Shape
    与张量类型的形状一致，描述数组的维度信息，由静态维度（整数）和动态维度（?）组成
- ElementType
    数组中单个元素的类型，可与张量的元素类型相同, 支持基本类型（如 i32、f64、index），复合类型（如 vector<4xi8>、struct<(i16, f32)>）
- Layout
    **strides（步长）**：一个整数列表，长度与数组维度相同，描述每个维度上 “跨一个元素” 需要跳过的内存单元数（以元素大小为单位）.
    **Offset（偏移量）**：整数，表示数组第一个元素在内存中的起始位置相对于 “基地址” 的偏移（以元素大小为单位）
    **affine_map 布局**
    通过仿射映射（Affine Map）自定义布局，支持更灵活的内存映射关系（如列优先、块划分等）。
    示例：memref<2x3xi32, affine_map<(i,j) -> (j*2 + i)>, 0> 定义了列优先的布局（元素 [i,j] 位置为 j×2 + i）。
- AddressSpace（地址空间）
    一个非负整数，用于区分不同的内存区域（如主机内存、GPU 全局内存、寄存器文件等），默认地址空间为 0。

```cpp
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
```

![alt text](./imgs/image.png)

###  向量类型
表示固定大小的一维或多维向量，通常用于 SIMD（单指令多数据）优化
格式：array<NxT>，N 为长度，T 为元素类型。
示例：array<5xi32> 表示含 5 个 i32 元素的数组。

```cpp
// 向量类型,定长的一段内存
auto vector_type = mlir::VectorType::get(3, f32);
llvm::outs() << "F32 1D向量类型 :\t";
vector_type.dump();

auto vector_2D_type = mlir::VectorType::get({3, 3}, f32);
llvm::outs() << "F32 2D向量类型 :\t";
vector_2D_type.dump();
```
![alt text](./imgs/image.png)

## 内建Attribute
属性是 MLIR IR 的 “标签”，用于：
- 配置操作行为（如卷积的kernel_size、循环的trip_count）
- 存储常量值（如arith.constant 10 : i32中的10）
- 关联 IR 符号（如调用函数时引用的@foo）
- 记录调试信息（如源文件的行号、列号）
核心特点是**静态性**—— 值在编译期确定，运行时不会改变，这是 MLIR 进行静态分析和优化的基础。

Attribute 具备一下特性
- 不可变性（Immutability）：属性创建后无法修改，若需更新只能创建新属性并替换旧值（保证 IR 线程安全和分析稳定性）。
- 可哈希性（Hashable）：属性支持哈希计算，MLIR 通过 “属性池（Attribute Pool）” 复用相同属性，减少内存开销。
- 类型化（Typed）：每个属性都有对应的AttributeType（如IntegerAttr的类型是IntegerType），确保类型安全。
- 可附加性（Attachability）：可附加到Operation（操作）、Block（块）、Module（模块）、Region（区域）等 IR 实体上。

### 常量值属性
直接存储编译期已知的具体值，是arith.constant等常量操作的核心元数据，也常用于操作的固定参数
#### IntegerAttr
任意位宽整数, 存储有符号（signed）或无符号（unsigned）整数，支持任意位宽.
描述方式： `[数值][: 类型标注]`, 类型标注格式为`i<位宽>`（有符号）或`ui<位宽>`（无符号），默认有符号
例如 `10 : i32` 表示 32 位有符号整数 10

#### FloatAttr
存储浮点数值，支持多种位宽（如 f16、f32、f64）和标准（如 IEEE 754）。
描述方式： `[数值][后缀][: 类型标注]`, 后缀h（半精度）、f（单精度）、无后缀（双精度）；类型标注为f<位宽>
例如 `3.14f : f32`（单精度浮点数 3.14）, `inf : f32`（单精度正无穷）、`nan : f64`（双精度非数）

#### BoolAttr
仅含true和false两个值，对应 MLIR 的i1类型（1 位整数）,直接写true或false，无需额外类型标注（类型固定为i1）

#### CharAttr
存储单个 Unicode 字符，对应 MLIR 的char类型。
描述方式： `'[字符]'[: 类型标注]`, 类型标注为char
例如 `'a' : char`（字符 'a'）, `'中' : char`（中文字符 '中'）

#### DenseIntOrFPElementsAttr
存储多维数组的密集整数或浮点元素，常用于张量或向量的常量初始化。
描述方式： `dense<[元素列表]> : 容器类型`, 若所有元素相同可简化为`dense<值>`
例如 `dense<[1, 2, 3] : i32>`（3 个 i32 元素的数组）, `dense<[1.0, 2.0, 3.0] : f32>`（3 个 f32 元素的数组）,`dense<[1,2,3,4]> : tensor<2x2xi32>`（2x2 整型张量，元素为 1、2、3、4）,`dense<0.0f> : tensor<4x4xf32>`（4x4 浮点张量，所有元素为 0.0）

#### SparseElementsAttr
存储多维数组的稀疏元素，适用于大部分元素为零的场景，节省内存。
描述方式： `sparse<[位置列表], [值列表]> : 容器类型`，位置用坐标表示（如 2D 张量的[行, 列]）
例如 `sparse<[[0,0],[1,2]], [10,20]> : tensor<3x3xi32>`（3x3 整型张量，位置(0,0)为10，(1,2)为20，其余为0）

### 复合属性
元数据需要包含多个子属性时（如操作的多参数配置），复合属性通过 “组合子属性” 实现结构化描述，避免属性数量膨胀。

#### ArrayAttr
存储有序属性列表，允许重复元素，类似于编程语言中的数组或列表。
描述方式： `[子属性1, 子属性2, ...] [: !array<子类型>]`，类型标注可选（可自动推断）
例如 `[3, 3] : !array<i32>`（卷积核大小：3x3）, `["conv2d", 64, true]`（混合类型：操作名、输出通道数、启用标志）

#### DictionaryAttr
存储无序、唯一键的键值对（键为字符串，值为任意属性），是描述操作复杂配置的 “标准格式”。
描述方式： `{键1: 值1, 键2: 值2, ...}`，键名唯一
例如 `{kernel_size: [3, 3], stride: [1, 1], padding: [0, 0]}`（卷积参数）, `{name: "relu", inplace: false}`（激活函数配置）

#### SetAttr
存储无序、唯一元素的集合，类似于编程语言中的集合或哈希集。
描述方式： `#set{子属性1, 子属性2, ...} [: !set<子类型>]`，类型标注可选
例如 `{1, 2, 3} : !set<i32>`（整数集合）, `{"conv2d", "relu"}`（字符串集合）

### 符号引用属性
用于引用 IR 中定义的符号（如函数、全局变量、类等），MLIR 中的 “符号（Symbol）” 是具有唯一名称的 IR 实体（如函数FuncOp、全局变量GlobalOp、模块Module），符号引用属性用于建立 IR 实体间的关联（如 “调用哪个函数”“访问哪个全局变量”）。
#### SymbolRefAttr
通用符号引用，引用任意层级的符号（支持嵌套符号，如嵌套模块中的函数），需指定完整符号路径。
描述方式： `@符号名`(全局符号)
例如 `@foo : func`（引用函数 @foo）, `@moduleA::submoduleB::bar`（引用嵌套模块 @moduleA 中的子模块 @submoduleB 中的函数 @bar）
用于跨层级的符号关联（如调用嵌套模块中的函数、访问嵌套全局变量）

#### FlatSymbolRefAttr
扁平符号引用， 仅引用**当前模块内的顶级符号**（不支持嵌套），查找效率比SymbolRefAttr更高（范围更小）
描述方式： `@符号名`
例如 `@global_var`（引用当前模块内的全局变量 @global_var）
用于模块内的符号关联（如调用当前模块内的函数、访问当前模块内的全局变量）

### 通用基础属性
MLIR IR的基础工具，常用的通用基础属性，覆盖最通用的元数据需求，所有方言都依赖它们。

#### UnitAttr
单元标记, 无具体值，仅作为 “存在性标记”—— 属性存在即表示某种语义，不存在则相反。
描述方式： 直接写属性名
例如 `block @entry attributes {is_entry_block} { ... }`（标记块entry为入口块）

#### StringAttr
字符串属性，存储任意长度的字符串，是最通用的文本元数据。
描述方式： `“字符串内容”`
例如 `"target = llvm.x86_64"`（目标平台标识）, `"comment = "一个注释""`

#### TypeAttr
引用类型， 直接引用 MLIR 中的一个 “类型（Type）”，用于描述与类型相关的参数（如类型转换的目标类型）。
描述方式： `<类型>`， 类型需符合 MLIR 的类型规范。
例如 `cast <f64>(%x) : (f32) -> f64`（cast操作的目标类型f64，由TypeAttr描述）

#### LocationAttr
位置属性， 描述 IR 元素的位置信息（如源文件路径、行号、列号），用于调试和错误报告。MLIR 要求每个Operation必须有LocationAttr
描述方式： 
基础位置：`loc("文件名", 行号:列号)`
合并位置：`loc(merge, 位置1, 位置2)`（多个源位置合并，如循环展开后的位置）
未知位置：`loc(unknown)`（无明确源位置时使用）