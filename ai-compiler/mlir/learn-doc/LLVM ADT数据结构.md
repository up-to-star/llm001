LLVM ADT（Abstract Data Types）容器是 LLVM 基础设施的核心组件，它们并非简单替代 STL，而是针对编译器场景的**极端性能敏感**、**内存分配敏感**、**迭代器稳定性敏感**等需求重新设计的。它的设计基于几个与编译器领域强相关的假设，这些假设决定了它们与STL的根本差异：

| 设计假设              | 对容器的影响                                  |
| --------------------- | --------------------------------------------- |
| **No Exceptions**     | 不处理异常安全，省去 try/catch 开销和回滚逻辑 |
| **No Allocators**     | 不支持自定义分配器模板参数，减少模板膨胀      |
| **No ABI Stability**  | 可以任意修改内部实现，不保证二进制兼容        |
| **Less Defensive**    | 不做前置下划线检查等防御性编程，更轻量        |
| **Internal Use Only** | 只为 LLVM 项目内部服务，不需要通用化          |

# 顺序容器

## `SmallVector<T, N>`

### 1. 核心设计：内联缓冲区（Inline Storage）

`SmallVector` 不是 `std::vector` 的简单包装，而是在对象内部**物理嵌入**了一段大小为 N 的未初始化存储空间。

```c++
// 简化后的内存布局示意
template<typename T, unsigned N>
class SmallVector {
    T      *Begin;           // 指向实际数据起始
    unsigned Size;           // 当前元素数
    unsigned Capacity;       // 当前容量（注意：Capacity >= N 恒成立）
    AlignedCharArrayUnion<T> InlineStorage[N]; // 内联缓冲区，与对象同生命周期
};
```

- 当 `Size <= N` 时，`Begin` 指向 `InlineStorage` 内部，零堆分配
- 当 `Size > N` 时，在堆上申请新内存，`Begin` 指向堆地址，`InlineStorage` 被闲置
- `Capacity` 的初始值就是 N，不是 0

### 2. 与 `std::vector<T>` 的逐项对比

| 维度                 | `SmallVector<T, N>`                                   | `std::vector<T>`                         |
| -------------------- | ----------------------------------------------------- | ---------------------------------------- |
| **对象大小**         | `sizeof(T*) + 2*sizeof(unsigned) + N*sizeof(T)`       | 通常 24 字节（三个指针）                 |
| **默认构造**         | 不触发堆分配                                          | 不触发堆分配（但容量为 0）               |
| **首次 `push_back`** | 若 `N >= 1`，直接在内联区构造，**零分配**             | 触发首次堆分配（通常申请 1 或 2 个元素） |
| **扩容策略**         | 与 `std::vector` 类似（通常 2 倍），但首次从 `N` 开始 | 通常 2 倍增长                            |
| **移动构造**         | 若数据在内联区，需逐元素移动；若在堆上，可窃取指针    | 始终窃取指针（三指针交换）               |
| **异常安全**         | **无**（LLVM 全局禁用异常）                           | 强异常安全保证                           |
| **自定义分配器**     | **不支持**                                            | 支持 `Allocator` 模板参数                |
| **迭代器类型**       | `T*`（原生指针）                                      | 通常是 `T*`（但标准不保证）              |

### 3. 工程陷阱：参数传递

**错误写法：**

```c++
void process(const SmallVector<int, 4>& vec);  // ❌ 灾难
```

因为 `SmallVector<int, 4>` 和 `SmallVector<int, 8>` 是完全不同的类型。调用方如果有一个 `SmallVector<int, 8>`，根本无法传入。

**正确写法：**

```c++
// 只读场景：用 ArrayRef（见下文）
void process(ArrayRef<int> vec);

// 需要修改的场景：用 SmallVectorImpl（与 N 无关的基类）
void process(SmallVectorImpl<int>& vec);
```

`SmallVectorImpl<T>` 是 `SmallVector<T, N>` 的基类，不包含 N 模板参数，因此可以接收任意 N 的 `SmallVector`。

## `ArrayRef<T>`

### 本质：非 owning 的连续内存视图

```c++
struct ArrayRef {
    T const* Data;
    size_t Length;
};
```

`ArrayRef<T>` 不拥有这块内存，也不管理其生命周期。它只是一个**轻量引用**（通常 16 字节），可以指向任何连续存储的序列。

与c++(20)的`std::span<T>`对比：
| 维度 | `ArrayRef<T>` | `std::span<T>` |
| --------------- | ------------------------------------------------- | ---------------------------------- |
| **出现时间** | LLVM 2.x 时代（~2008） | C++20（2020） |
| **元素可变性** | 始终 `const T`（只读视图） | 支持 `T` 和 `const T` 两种 |
| **静态大小** | 不支持 | 支持 `std::span<T, N>` |
| **切片 API** | `.slice(n, m)`, `.drop_front(n)`, `.take_back(n)` | `.subspan(pos, count)` |
| **空状态** | `empty()` + `data() == nullptr` | `empty()`（不保证 `data() != nullptr`） |
| **与 LLVM 生态集成** | 原生支持 `StringRef`、`SmallVector` 隐式转换 | 需适配 |

### 生命周期陷阱（Critical）

`ArrayRef<T>` 是悬挂引用（Dangling）的重灾区。

```c++
ArrayRef<int> getData() {
    SmallVector<int, 4> vec = {1, 2, 3};
    return vec;  // ❌ vec 析构，内联存储失效，ArrayRef 指向垃圾
}

ArrayRef<int> bad = getData();  // UB
```

```c++
ArrayRef<int> ref = someVector;
someVector.push_back(42);       // 若触发扩容，someVector 的 Data 指针改变，ref 悬挂
```

### 与 `SmallVector<T, N>` 的协同关系

`SmallVector<T, N>` 可以隐式转换为 `ArrayRef<T>`：

```c++
SmallVector<int, 4> vec = {1, 2, 3};
ArrayRef<int> ref = vec;  // 隐式转换，零开销
```

这构成了 LLVM 的参数传递规范：

- 函数输入（只读）：`ArrayRef<T>`
- 函数输出/可修改输入：`SmallVectorImpl<T>&` 或 `SmallVector<T, N>&`（仅在 N 确定时）

# 字符串容器

## `StringRef`

本质是非 owning 的字符串视图，类似 c++ 的 `std::string_view`。

```c++
class StringRef {
    const char *Data;      // 指向字符串起始（不保证 '\0' 结尾）
    size_t      Length;    // 显式长度
};
```

`StringRef` 只持有指针 + 长度，不管理内存。它通过显式长度支持内部含 `\0` 的字符串（如 `"foo\0bar"` 长度为 7），这是与 C 风格 `const char*` 的本质区别。

| 维度                 | `StringRef`                                                  | `std::string_view`                                |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| **出现时间**         | LLVM 2.x（~2008）                                            | C++17（2017）                                     |
| **空状态**           | `Data == nullptr && Length == 0`                             | `Data` 可能非空（标准不保证 `data() == nullptr`） |
| **转 `const char*`** | **不能隐式转换**（因为不一定 `\0` 结尾）                     | 同左（`data()` 不保证 null-terminated）           |
| **哈希支持**         | 内置 `DenseMapInfo` 特化，可直接做 `DenseMap` 键             | 标准提供 `std::hash`，但需适配 LLVM 容器          |
| **切片 API**         | `.slice(n, m)`, `.split(Sep)`, `.trim()`, `.consume_front()` | `.substr(pos, count)`                             |
| **与 LLVM 生态集成** | 与 `Twine`、`StringMap`、`raw_ostream` 无缝协作              | 需手动桥接                                        |

**核心API**

```c++
StringRef s = "  hello,world  ";

// 查询
s.size();              // 15
s.empty();             // false
s.startswith("  he"); // true
s.contains("lo,wo");   // true
s.find(',');           // 8

// 切片（均返回新的 StringRef，零拷贝）
s.slice(2, 5);         // "hel"（从 2 开始，长度 5）
s.drop_front(2);       // "hello,world  "（去掉前 2 个字符）
s.drop_back(2);        // "  hello,worl"（去掉后 2 个字符）
s.trim();              // "hello,world"（去掉两端空白）

// 分割
auto [lhs, rhs] = s.split(',');  // lhs="  hello", rhs="world  "

// 消费前缀（常用于解析器）
StringRef input = "int x;";
StringRef keyword;
if (input.consume_front("int "))   // 若前缀匹配则截断，返回 true
    keyword = "int";
```

## Twine —— 延迟字符串拼接

**1. 核心设计：表达式树，零临时分配**
Twine 是 LLVM 中最独特的字符串容器。它不存储字符串内容，而是记录"如何拼接"的操作树。

```c++
class Twine {
    // 二叉树节点：左操作数 + 右操作数 + 操作类型
    const Node *LHS;
    const Node *RHS;
    NodeKind    Kind;  // TwineKind, CStringKind, StdStringKind, etc.
};

Twine msg = Twine("Error in ") + FuncName + ": " + ErrMsg;
```

此时零分配！Twine 只是构建了一棵树：

```c++
            +
          /   \
        "Error in "   +
                    /   \
                FuncName  +
                          / \
                        ": " ErrMsg
```

## `SmallString<N>`

**1. 核心设计：SmallVector<char, N> 的特化**

```c++
template<unsigned N>
class SmallString : public SmallVector<char, N> {
    // 继承 SmallVector 的全部机制
    // 额外提供字符串专用 API
};
```

| 维度                    | `SmallString<N>`                       | `std::string`                     |
| ----------------------- | -------------------------------------- | --------------------------------- |
| **SSO（短字符串优化）** | 显式控制阈值 `N`                       | 实现定义（通常 15~22 字节）       |
| **对象大小**            | 较大（24 + N 字节）                    | 较小（通常 32 字节含 SSO）        |
| **堆分配策略**          | 与 `SmallVector` 一致（2 倍扩容）      | 实现定义                          |
| **接口丰富度**          | 基础（`append()`、`c_str()`、`str()`） | 极丰富                            |
| **与 StringRef 互转**   | `StringRef(SmallString)` 隐式转换      | `StringRef(std::string)` 隐式转换 |
| **异常安全**            | 无                                     | 强                                |

# 集合容器

SmallPtrSet<T, N>_ —— 对标 std::set<T_> / std::unordered_set<T*>
核心机制：小尺寸时用线性数组（T* SmallStorage[N]），大尺寸时切换为二次探测哈希表 。
与 STL 对比：

| 特性         | `SmallPtrSet<T*, N>` | `std::unordered_set<T*>` | `std::set<T*>` |
| ------------ | -------------------- | ------------------------ | -------------- |
| 小尺寸存储   | 内联数组，零分配     | 始终桶分配               | 树节点分配     |
| 探测策略     | 二次探测（开放寻址） | 链地址法                 | 红黑树         |
| 迭代器稳定性 | 插入/删除会失效      | 插入可能失效             | 删除仅当前失效 |
| 遍历顺序     | 无序                 | 无序                     | 有序           |
| 内存局部性   | 极佳（单一连续存储） | 差（节点分散）           | 差（节点分散） |

> 为什么不用 `std::unordered_set`：LLVM 官方文档明确说 "We never use containers like unordered_map because they are generally very expensive (each insertion requires a malloc)" 。

## `DenseSet<T>` / `DenseMap<K, V>`

**核心机制**：扁平化开放寻址哈希表，二次探测，单一连续内存块。

```cpp
// 内部结构示意
template<typename KeyT, typename ValueT>
class DenseMap {
    BucketT *Buckets;      // 连续数组
    unsigned NumEntries;
    unsigned NumTombstones;  // 删除标记（开放寻址必需）
};
```

与 `std::unordered_map` 的深层对比：

| 特性         | `DenseMap`                                    | `std::unordered_map`     |
| ------------ | --------------------------------------------- | ------------------------ |
| 存储结构     | 扁平数组，开放寻址                            | 桶数组 + 链表/树         |
| 缓存友好性   | 极高（键值对连续存储）                        | 差（节点离散）           |
| 负载因子控制 | 默认 0.75，自动 rehash                        | 同左                     |
| 迭代器失效   | 任何插入都失效                                | 仅 rehash 时失效         |
| 空桶标记     | 需要 DenseMapInfo 提供两个哨兵值（空桶/墓碑） | 不需要                   |
| 自定义键     | 需特化 `DenseMapInfo<KeyT>`                   | 需特化 `std::hash<KeyT>` |
| 内存分配次数 | 极少（单一数组）                              | 多（桶数组 + 每个节点）  |

**关键限制**：

- 键类型必须提供 `DenseMapInfo` 特化，定义空桶和墓碑值（如指针可用 `nullptr` 和 `-1` 指针）
- 不适合存储大对象（初始默认分配 64 个桶，大对象会浪费空间）

## `SparseSet<T>`

**无直接 STL 对标**。专为编译器场景设计：键是 `unsigned` 类型的密集整数（如物理寄存器编号、虚拟寄存器 ID）。

**机制**：

- 用一个大数组 `Sparse` 做索引，一个小数组 `Dense` 存实际数据
- `Sparse[Key] = IndexInDense`，实现 O(1) 查找
- 清空操作只需重置 Dense 大小，O(1)

**适用场景**：寄存器分配、活跃变量分析等"键空间很大但实际元素很少"的场景。

## `StringMap<V>`

**对标** `std::unordered_map<std::string, V>`

**优化点**：键存储为 `StringRef`（长度 + 指针），但内部持有字符串拷贝。针对字符串键做了内存布局优化，减少分配碎片。

## `IndexedMap<T>`

**对标** `std::vector<T>` 的语义化包装

**机制**：底层是 `std::vector<T>`，但通过映射函数将键（如虚拟寄存器 ID）映射到密集索引。

```cpp
// 虚拟寄存器编号可能从 1000 开始，IndexedMap 将其映射到 0..N-1
IndexedMap<RegInfo> VirtRegMap;
```

## `MapVector<K, V>`

**对标** `std::map` + 插入顺序保持

**机制**：`DenseMap<K, unsigned>` 映射到 `std::vector<std::pair<K, V>>` 的索引。

**特点**：

- 保证迭代顺序 = 插入顺序（解决指针键的遍历不确定性）
- 键存储两份（`DenseMap` 一份 + `Vector` 一份）
- 删除是 O(N)（需移动 vector），建议批量 `remove_if()`

**为什么需要它**：LLVM 中很多 Pass 用指针做键，普通 `DenseMap` 遍历顺序不确定，导致输出 IR 在不同运行间不一致（影响调试和测试）。

## `BitVector` / `SparseBitVector`

| 容器              | 对标 STL                            | 特点                                           |
| ----------------- | ----------------------------------- | ---------------------------------------------- |
| `BitVector`       | `std::vector<bool>` / `std::bitset` | 动态大小，字级别操作，比 `vector<bool>` 更高效 |
| `SmallBitVector`  | -                                   | 小尺寸时内联存储                               |
| `SparseBitVector` | -                                   | 稀疏位集，用链表存储非零字，适合活跃变量分析   |

# 三、LLVM 重新设计容器的根本原因

综合官方文档和社区讨论，LLVM 不使用 STL 容器的核心原因如下：

## 1. 内存分配极致优化

编译器是分配密集型应用。解析一个大型 C++ 文件可能产生数百万个 AST 节点和 IR 指令。`std::vector` 每次默认构造都要堆分配，而 `SmallVector<4>` 可以覆盖 90% 的实际使用场景（如函数参数列表通常很短）零分配。

## 2. 缓存局部性（Cache Locality）

`DenseMap` 的开放寻址将所有键值对放在连续数组中，CPU 缓存命中率远高于 `std::unordered_map` 的离散节点。在 Pass 遍历海量 IR 元数据时，这直接决定编译速度。

## 3. 侵入式数据结构

`ilist` 的侵入式设计让 `Instruction` 对象本身就能链接成链表，避免每个指令额外分配一个链表节点。这在 IR 层面节省了数倍的内存。

## 4. 控制迭代器失效语义

LLVM 明确选择更激进的迭代器失效策略（如 `DenseMap` 插入即失效），换取更简单的内部实现和更高性能。STL 为了保证迭代器稳定性付出了巨大复杂度代价（如 `std::list` 的节点不移动策略）。

## 5. 避免模板膨胀和 ABI 约束

STL 容器是模板，每个实例化都生成独立代码。LLVM ADT 通过 `SmallVectorImpl<T>`（与 N 无关的基类）等手段减少代码膨胀。且 LLVM 不承诺 ABI 稳定，可以随版本任意优化内部布局。

## 6. 编译器领域的特殊假设

- 键类型常为指针或整数，适合哈希
- 集合大小通常可预测（如基本块的指令数）
- 需要确定性迭代顺序（用于测试和调试）
- 不需要异常安全
