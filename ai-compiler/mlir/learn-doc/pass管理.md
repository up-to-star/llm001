## Pass是什么

编译器的核心工作是把高层表示逐步转换成低层表示，同时做各种优化。“转换和优化”的过程不可能一步完成，需要拆成很多小步骤，每个步骤专注做一件事。
在MLIR中，每个步骤被称为一个Pass。

一个Pass就是对IR的一次独立处理，他可以做变换，比如把一种操作替换成另一种操作；也可以做优化，比如消除冗余计算、内联函数。
**Pass的设计原则是“单一职责原则”**。每个Pass只关心一件事，多个Pass组合起来完成复杂的编译流程。
每个 Pass 做的事情可能很小，但组合起来就能完成从高层抽象到机器码的整个转换链。
例如：

```mlir
func.func @example(%arg0: i32) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32    // %arg0 + %arg0
  %1 = arith.muli %0, %arg0 : i32       // 上一步结果 * %arg0
  %2 = arith.addi %arg0, %arg0 : i32    // 又算了一遍 %arg0 + %arg0
  %3 = arith.addi %0, %2 : i32          // %0 和 %2 其实是一样的
  return %3 : i32
}
```

这里 `%0` 和 `%2` 计算的是完全相同的东西。一个叫 CSE（Common Subexpression Elimination，公共子表达式消除）的 Pass 可以发现这个冗余，把 `%2` 的使用全部替换成 `%0`，然后删掉 `%2`。

## Pass处理的对象

Pass处理的对象是IR中的操作（Operation）。每个操作都有一个IR表示，比如 `arith.addi`、`arith.muli` 等。甚至整个 `Module` 也是一个 Operation（`builtin.module`）。
**MLIR 的 IR 结构**:

```mlir
Operation
├── Region（区域）
│   ├── Block（基本块）
│   │   ├── Operation
│   │   ├── Operation
│   │   └── ...
│   └── Block
│       └── ...
└── Region
    └── ...

```

Operation 可以包含 Region，Region 包含 Block，Block 里面又是 Operation。这是一个递归嵌套的结构。
Pass 的工作方式是：选定一个 Operation 作为”根”，然后处理这个 Operation 及其内部的子树。比如一个作用在 func.func 上的 Pass，它只能看到和修改函数内部的东西，不能去动函数外面的 IR。

### Pass并行安全的关键

MLIR支持多线程执行Pass。例如一个Module里有10个函数，MLIR可以同时在不同的线程上对这10个函数跑同一个Pass。
但是会带来一个问题：如果函数A和函数B在不同线程上同时被处理，它们不能互相干扰。解决这个问题的关键是：`IsolatedFromAbove`这个**trait**（特性标记）。
一个Operation如果带有`IsolatedFromAbove` trait，意味着它的内部区域不会引用额外定义的SSA值，也就是说它是“自包含”的，内部不依赖外部的东西。

**关键约束**：只有带 IsolatedFromAbove trait 的 Operation 才能作为 Pass 的调度目标。这是 MLIR Pass 基础设施的硬性要求。
当 Pass Manager 要在某个 Operation 上运行 Pass 时，会先检查：

```c++
if (!opInfo->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
  return op->emitOpError()
         << "trying to schedule a pass on an operation not marked as 'IsolatedFromAbove'";
}
```

常见的带 `IsolatedFromAbove` trait 的 Operation 包括：

- `builtin.module`（模块）
- `func.func`（函数）
- `gpu.module`（GPU 模块）
- `gpu.func`（GPU 函数）

值得一提的是，`IsolatedFromAbove`的检查发生在运行时对具体 Operation 调度前，而非管线 finalize 阶段。finalize 阶段只校验”pass 与锚定类型是否匹配”，不会提前检查具体 op 的 trait。

`finalize` 是 PassManager 在真正开始遍历 IR 之前的一个验证/准备阶段。
具体来说， MLIR 的 PassManager::run(ModuleOp) 内部大致分为三个阶段：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   1. 构建阶段    │ ──→ │  2. Finalize   │ ──→ │  3. 运行时执行   │
│  (Parsing/Add)  │     │   (验证/准备)    │     │  (遍历IR调度)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

| 阶段         | 时机                                            | 做什么                                          |
| ------------ | ----------------------------------------------- | ----------------------------------------------- |
| **构建**     | `pm.addPass()` 或 `parsePassPipeline()` 时      | 构建 `OpPassManager` 的树形结构                 |
| **Finalize** | `pm.run(module)` 被调用后，**实际遍历 IR 之前** | 校验 Pass 与锚定类型的匹配性，做执行前准备      |
| **运行时**   | Finalize 通过后                                 | 递归遍历 IR，对每个 Operation 判断是否执行 Pass |

**Finalize 阶段具体做什么？**
"finalize 阶段只校验 pass 与锚定类型是否匹配", 锚定类型（Anchor Type）指的是 OpPassManager 所挂载的 Operation 类型。
例如：

```c++
// 伪代码示意 PassManager::run 内部逻辑
LogicalResult PassManager::run(Operation *op) {
  // ====== 阶段 1: Finalize ======
  // 检查：这个 PassManager 里的每个 Pass，能否合法地挂载在当前 op 类型上
  for (auto *pass : passes) {
    // 只检查：OperationPass<FuncOp> 的 Pass 是否被挂到了 FuncOp 层级？
    // 只检查：OperationPass<ModuleOp> 的 Pass 是否被挂到了 ModuleOp 层级？
    if (!canPassRunOnOp(pass, op->getName())) {
      return failure();  // "pass 与锚定类型不匹配"
    }
  }

  // ====== 阶段 2: 运行时执行 ======
  // 递归遍历 IR，调度 Pass
  return executePipeline(op);
}
```

```c++
PassManager pm(module.getName());
pm.addPass(std::make_unique<<MyFuncPass>>());  // MyFuncPass : OperationPass<FuncOp>

// Finalize 检查：
// "MyFuncPass 是 OperationPass<FuncOp>，但 PassManager 锚定在 ModuleOp 上"
// → 报错：pass 与锚定类型不匹配
```

**为什么 `IsolatedFromAbove` 要在运行时检查？**
`IsolatedFromAbove` 是一个 Operation Trait，表示：

> "这个 Operation 内部的 Region 与外部没有隐式的 SSA 值依赖（除了显式声明的 Block Arguments）"

**关键原因**：Finalize 阶段只看到了 `PassManager` 的结构（管线的嵌套层级），但还没有看到 具体的 IR 实例。一个 Operation 是否实现了 `IsolatedFromAbove`，是具体 Operation 实例的属性，不是 `PassManager` 结构层面的信息。

## 三种Pass

根据 Pass 作用的 Operation 类型不同，MLIR 提供了三种 Pass 基类：

| 类型     | 基类                                   | 作用范围                         |
| -------- | -------------------------------------- | -------------------------------- |
| 特定 Op  | OperationPass<OpT>                     | 只能在指定类型的 Op 上运行       |
| Op 无关  | OperationPass<void> 或 OperationPass<> | 可以在任意 Op 上运行             |
| 接口过滤 | InterfacePass<InterfaceT>              | 只能在实现了指定接口的 Op 上运行 |

特定 Op 的 Pass 最常见，比如只作用在 func.func 上的函数级优化：

```c++
struct MyFuncPass : public OperationPass<FuncOp> {
  MyFuncPass() : OperationPass<FuncOp>() {}
};
```

Op 无关的 Pass 更通用，可以被放到任何 Pass Manager 里。CSE 和 Canonicalize 就是这类 Pass，它们不关心具体是什么 Op，只要有东西可以处理就行：

```c++
struct MyAgnosticPass : public OperationPass<> {
  void runOnOperation() override {
    Operation *op = getOperation();  // 通用的 Operation*
    // 不能假设 op 是什么具体类型
  }
};
```

接口过滤的 Pass 介于两者之间。它不限定具体 Op 类型，但要求 Op 必须实现某个接口。比如只作用在实现了 FunctionOpInterface 接口的 Op 上：

```c++
struct MyInterfacePass : public InterfacePass<FunctionOpInterface> {
  void runOnOperation() override {
    FunctionOpInterface op = getOperation();  // 可以用接口方法
    // 适用于所有"函数式"的 Op，不只是 func.func
  }
};
```

### 写Pass注意的点

- **不能读兄弟 Op 的状态。** 当前 Op 运行时，不能读取当前 Op 的兄弟节点可能正在被其他线程修改。
- **只能改当前 Op 子树内的东西。** 不能去修改父级 Block 里的其他 Op，也不能往父级 Block 里插入新 Op。唯一的例外是可以修改当前 Op 自己的 attributes。
- **不能在 runOnOperation 调用之间保持可变状态。** Pass 实例可能被克隆到多个线程，不保证执行顺序，也不保证每个实例都会处理所有 Op。
- **Pass 必须可拷贝。** Pass Manager 会克隆 Pass 实例来并行处理。
- **要声明依赖的 Dialect。** 如果 Pass 会创建新的 Operation、Type 或 Attribute，需要在 `getDependentDialects()` 里声明这些实体所属的 Dialect（方言）。Dialect 是 MLIR 中组织相关 Operation、Type、Attribute 的命名空间。

## PassManager 与管线组织

PassManager是将多个Pass组织起来的类，它负责在IR上执行这些Pass。
MLIR的Pass管理有两个核心类：

- `PassManager`：顶层入口，负责整个管线的配置和执行。
- `OpPassManager`：管理在某一层级上运行的 Pass 集合

PassManager 本身继承自 OpPassManager，所以它既是入口也是一个 Pass 容器。

OpPassManager 有一个重要概念叫”锚定”（anchor）。每个 OpPassManager 都锚定到某种 Operation 上，表示这个管理器里的 Pass 会在什么类型的 Op 上执行。锚定有两种模式：
| 模式 | 含义 | 用法 |
|------|------|------|
| op-specific | 锚定到特定 Op 类型 | OpPassManager("func.func") |
| op-agnostic | 不限定 Op 类型，可以在任意可行 Op 上运行 | OpPassManager() 或锚定到 "any" |

### 嵌套管线

MLIR 的 IR 是嵌套结构的，Pass 管线也是嵌套的。
例如：

```mlir
module {
  spirv.module Logical GLSL450 {
    spirv.func @compute() {
      // ...
    }
  }
  func.func @host() {
    // ...
  }
}

builtin.module
├── spirv.module
│   └── spirv.func
└── func.func
```

如果想在不同层级上运行不同的 Pass，需要构建嵌套的管线：

```c++
// 创建顶层 PassManager，锚定到 ModuleOp
auto pm = PassManager::on<ModuleOp>(ctx);

// 在 module 层添加一个 pass
pm.addPass(createMyModulePass());

// 嵌套一个 spirv.module 层级的管理器
OpPassManager &spirvPM = pm.nest<spirv::ModuleOp>();
spirvPM.addPass(createSPIRVModulePass());

// 再嵌套一个 func.func 层级的管理器
OpPassManager &funcPM = pm.nest<func::FuncOp>();
funcPM.addPass(createCanonicalizerPass());
funcPM.addPass(createCSEPass());

// 执行
if (failed(pm.run(module)))
  // 处理失败...
```

这段代码构建的管线结构是：

```
OpPassManager<builtin.module>
├── MyModulePass
├── OpPassManager<spirv.module>
│   └── SPIRVModulePass
└── OpPassManager<func.func>
    ├── Canonicalizer
    └── CSE
```

### 嵌套模式：Explicit vs Implicit

往 OpPassManager 里添加 Pass 时，有个问题：如果 Pass 要求的 Op 类型和 OpPassManager 的锚定类型不一致怎么办？

这由 Nesting 枚举控制：

- Explicit 模式（默认）：Pass 的目标类型必须和 OpPassManager 的锚定一致，否则报错。
- Implicit 模式：如果类型不一致，自动创建一个嵌套的子 OpPassManager 来容纳这个 Pass。

### OpToOpPassAdaptor：嵌套执行的幕后推手

你可能会好奇：嵌套的 OpPassManager 是怎么实际执行的？

答案是 OpToOpPassAdaptor（见 PassDetail.h）。当你调用 pm.nest<SomeOp>() 时，实际上创建了一个 OpToOpPassAdaptor，它是一个特殊的 Pass，负责：

- 遍历当前 Op 的所有 Region 和 Block
- 找到每个匹配锚定类型的子 Op
- 在这些子 Op 上执行嵌套的管线

### Adaptor 合并优化

如果连续添加多个指向同一层级的嵌套 Pass，MLIR 会自动合并 Adaptor，避免重复遍历。

比如：

```c++
pm.nest<func::FuncOp>().addPass(createCSEPass());
pm.nest<func::FuncOp>().addPass(createCanonicalizerPass());
```

这会产生两个独立的 Adaptor。但在 finalizePassList 阶段，MLIR 会把它们合并成一个：

```
// 合并前
Adaptor1 -> [CSE]
Adaptor2 -> [Canonicalizer]

// 合并后
Adaptor -> [CSE, Canonicalizer]
这样只需要遍历一次 IR 结构就能执行两个 Pass。
```

不过合并也有限制：若相邻 adaptor 中存在锚定为 any 的通用 PassManager，且与其它非通用 PassManager 存在潜在的可调度冲突，出于保守原则将不进行合并。在可合并的情形下，合并后会对内部的 PassManager 按锚定排序——先列出锚定到具体 Op 的（按 Op 名字字典序），最后才是 any。

## Pass 执行顺序与调度

理解 Pass 是怎么被执行的，对写出正确的 Pass、调试管线问题都很关键。这一部分来拆解 MLIR 的执行模型。

**PassManager::run 做了什么**
当你调用 pm.run(module) 时，实际发生了这些事情：

- 检查锚定匹配 → 确认目标 Op 类型和 PM 锚定一致
- 加载依赖 Dialect → 遍历所有 Pass 的 `getDependentDialects()`，预加载
- finalize 管线 → 合并相邻 Adaptor、校验可调度性
- 进入多线程域 → `context->enterMultiThreadedExecution()`
- 初始化 Pass → 调用每个 Pass 的 `initialize()`（如果需要）
- 初始化代次判断 → 通过“dialect registry 哈希 + pipeline 哈希”判断是否需要新一轮 `initialize`，以复用上次初始化结果（若管线/注册表未变化）
- 构建 AnalysisManager
- 执行管线 → `OpToOpPassAdaptor::runPipeline()`
- 退出多线程域 → `context->exitMultiThreadedExecution()`
- 输出统计信息（如果开启）

`runPipeline`实现要点：

```c++
LogicalResult OpToOpPassAdaptor::runPipeline(
    OpPassManager &pm, Operation *op, AnalysisManager am, ...) {

  // 触发 instrumentation 钩子
  if (instrumentor)
    instrumentor->runBeforePipeline(...);

  // 按顺序执行每个 Pass
  for (Pass &pass : pm.getPasses()) {
    if (failed(run(&pass, op, am, verifyPasses, parentInitGeneration)))
      return failure();
  }

  if (instrumentor)
    instrumentor->runAfterPipeline(...);

  return success();
}
```

很直接：遍历 Pass 列表，按加入顺序依次执行。任何一个 Pass 失败，整个管线立即终止。

### 同层 vs 嵌套的执行顺序

假设有这样的 IR：

```mlir
module {
  func.func @foo() { ... }
  func.func @bar() { ... }
}
```

和这样的管线：

```
builtin.module(
  module-pass-A,
  func.func(func-pass-1, func-pass-2),
  module-pass-B
)
```

执行顺序是：

```
1. module-pass-A    on module
2. func-pass-1      on @foo
3. func-pass-2      on @foo
4. func-pass-1      on @bar
5. func-pass-2      on @bar
6. module-pass-B    on module
```

- 同层 Pass 按加入顺序执行：先 A，然后嵌套管线，最后 B
- 嵌套管线处理完所有子 Op 后才返回：func-pass-1 和 func-pass-2 在 @foo 和 @bar 上都执行完，才轮到 module-pass-B
- 每个子 Op 上，嵌套管线内的 Pass 按顺序执行：@foo 上先跑 func-pass-1 再跑 func-pass-2，然后 @bar 上同样顺序

这种设计的好处是局部性好。处理 @foo 时，相关数据都在缓存里；处理完 @foo 的所有 Pass 再去处理 @bar，比”所有函数跑 pass-1，再所有函数跑 pass-2”的缓存命中率高。

### 多线程执行

MLIR 支持并行执行 Pass，但有严格的规则。

并行发生在哪里？

同一个父 Op 下的多个子 Op 可以并行处理。上面的例子里，@foo 和 @bar 可以同时处理：

```
时间 →
module: [A]─────────────────[B]
@foo: [1]──[2]
@bar: [1]──[2] ← 和 @foo 并行
```

并行不发生在哪里？
同一个 Op 上的多个 Pass 不并行，必须按顺序
父子层级之间不并行，父层 Pass 执行完才进入子层

## 代表性 Pass

### Canonicalize：规范化 IR

Canonicalize 是 MLIR 里用得最多的 Pass 之一。它的作用是把 IR 变换成”规范形式”，让后续的 Pass 更容易匹配和优化。

它做的事情包括：

- 消除无副作用且无使用者的操作
- 常量折叠，比如把 arith.addi 1, 2 折成 3
- 把常量操作数移到右边，比如 arith.addi 4, x 变成 arith.addi x, 4
- 把 constant-like 操作提升到入口块
- 各种 dialect 自定义的规范化规则

  Canonicalize 基于 Greedy Pattern Rewrite Driver。它不断尝试匹配和应用各种 pattern，直到 IR 不再变化（达到不动点）或者达到最大迭代次数。

### CSE：公共子表达式消除

CSE（Common Subexpression Elimination）找出等价的计算，把重复的删掉。

来看个例子：

```
// 变换前
%0 = arith.addi %a, %b : i32
%1 = arith.addi %a, %b : i32  // 和 %0 完全一样
%2 = arith.muli %0, %1 : i32

// CSE 之后
%0 = arith.addi %a, %b : i32
%2 = arith.muli %0, %0 : i32  // %1 被替换成 %0
```

CSE 的核心逻辑是基于支配树（Dominance Tree）遍历。它用一个 scoped hash table 记录已经见过的操作，遇到等价的就替换。
CSE 有个特点：它不会删除带 Region 的操作（比如 scf.for、func.func）。这意味着 CFG 结构不变，所以可以保留 DominanceInfo 和 PostDominanceInfo。

**Greedy vs Walk 驱动**
Canonicalize 和 CSE 背后用的是不同的重写驱动。

- Greedy Pattern Rewrite Driver：
  Canonicalize 用的就是这个
  维护一个 worklist，不断尝试 pattern
  修改/新建的操作会被加回 worklist
  迭代到不动点或达到最大次数
  支持 top-down 和 bottom-up 遍历模式
- Walk Pattern Rewrite Driver：
  简单的后序遍历
  按 pattern benefit 排序尝试
  不回访已修改的操作
  更快，但功能弱一些
  如果你的变换需要迭代到不动点，用 Greedy。如果只是简单的一遍扫描，用 Walk。

经典组合：canonicalize → cse → canonicalize
第一轮 canonicalize：规范化 IR，暴露更多可消除的冗余
cse：消除公共子表达式
第二轮 canonicalize：CSE 可能创建新的优化机会，再清理一遍
