参考原文：https://zhuanlan.zhihu.com/p/1984363433035052836

# 为什么需要MemRef Dialect

编译器看到多维数组时，难以分析数据的形状、布局等信息。例如：

```c++
void matmul(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
```

这段代码人类看起来很清楚，但编译器会遇到以下几个问题：

- **形状信息丢失**
  编译器拿到的是`float *A`，它只知道这是个指针，指向一段连续内存，至于这段内存是`MxK`的二维数组还是`M×K×1`的三维数组，或者就是一个展平的一维数组，编译器完全不知道。形状信息在函数签名里就消失了，只能靠程序员在循环体力用`i * K + k`这样的表达式来暗示。
- **布局方式不确定**
  代码中：

  ```c++
  C[i * N + j] = ...  // 行主序，stride 是 N
  ```

  这是行主序的访问模式。但如果数组是列主序存储的呢？表达式就得改成 C[j * M + i]。编译器拿到 IR 之后，很难判断这个数组到底是怎么排列的。
  此外数组可能有 padding，可能是 strided 的，可能有 offset。编译器想做循环向量化、循环重排、数据预取这些优化，但缺少布局信息，很多优化根本不敢做。

- **别名分析困难**
  例如：

  ```c++
  void foo(float *A, float *B) {
      A[0] = 1.0f;
      B[0] = 2.0f;
      float x = A[0];  // 这里 x 是 1.0 还是 2.0？
  }
  ```

  如果A和B指向同一块内存，那么x就是2.0。编译器需要做别名分析，但指针类型提供的信息太少了。编译器会保守处理，放弃优化机会。

鉴于这些问题，MLIR引入了MemRef类型，把丢失的信息重新编码到类型系统里。
例如：

```c++
func.func @simple_access(%A: memref<4x8xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %val = memref.load %A[%c0, %c1] : memref<4x8xf32>
    return %val : f32
}
```

这段 MLIR 代码里，`memref<4x8xf32>` 就是 MemRef 类型。它直接在类型里声明了：

- 这是个二维数组（两个维度）
- 第一维大小是 4
- 第二维大小是 8
- 元素类型是 f32

编译器一看到这个类型，立刻就知道了数组的完整结构。访问 %A[%c0, %c1] 时，编译器能直接算出线性地址，不需要猜测。

# MemRef 类型结构

MemRef 类型的通用形式是这样的：

```c++
memref<shape x element_type, layout, memory_space>
```

- shape部分
  可以是静态的数字，也可以是动态的 ?：
  - `memref<4x8xf32>` 表示静态 4×8 数组
  - `memref<?x?xf32>` 表示完全动态的二维数组
  - `memref<4x?xf32>` 表示混合的，第一维静态，第二维动态

- element_type部分
  可以是任意 MLIR 类型，例如 `f32`、`i32`、`bf16` 等。`vector<4xf32>` 甚至可以是向量类型。

- layout部分
  layout 描述了多维索引如何映射到线性内存地址。
  - 最简单的是省略 layout，默认就是 identity layout，也就是标准的行主序：

    ```c++
    memref<4x8xf32>
    ```

    这表示 4×8 数组，每个元素是 f32，行主序存储。访问[i, j] 时，地址是 i \* 8 + j。

  - 如果需要更复杂的布局，可以用 strided layout：

    ```c++
    memref<12x4xf32, strided<[4, 1], offset: 5>>
    ```

    这表示 12×4 数组，每个元素是 f32，行主序存储，第一维的stride 是 4，第二维的stride 是 1，偏移量是 5。
    访问[i, j] 时，地址是 (i \* 4 + j \* 1) + 5。

  - 更通用的方式是用 affine map：

    ```c++
    memref<8x64xf32, affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>, 1>
    ```

    这个 affine map 显式定义了索引映射关系。d0, d1 是维度索引，s0 是符号参数，可以在运行时绑定。

- memory_space部分
  可以是 MLIR 内存空间，例如 `memref<4x8xf32, memory_space: 0>` 可能表示在内存空间 0 中分配内存。
  这里的数字只是标签，具体哪一个代表 CPU/GPU/某级缓存，由目标和编译管线约定。MemRef 类型只是把这个“内存空间标签”写进类型里，后端可以根据它选择合适的访问/传输实现。

## 带 affine map 的示例

假设有一个更大的逻辑矩阵 Base，有 32 行、N 列，我们希望每次只处理其中连续的 8 行：
第 0 个窗口：行 [0..7]
第 1 个窗口：行 [8..15]
第 2 个窗口：行 [16..23]
第 3 个窗口：行 [24..31]

对于第 b 个窗口，我们可以用一个带 affine map 的 MemRef 类型来描述“以第 row_offset = b \* 8 行为起点的 8×N 子矩阵”，形状是 8x?，偏移由符号参数控制：

```c++
// cols: 当前矩阵的列数 N
// row_offset: 这个 8 行窗口在全局矩阵中的起始行号
%tile = memref.alloc(%cols)[%row_offset]
          : memref<8x?xf32,
                   affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>, 1>

// 后续可以像普通 8xN 矩阵一样访问 %tile
%v = memref.load %tile[%i, %j]
       : memref<8x?xf32,
                affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>, 1>
```

其中：

- `(%cols)` 绑定到类型中的动态维度 `?`，也就是列数 `N`；
- `%row_offset` 绑定到 affine map 里的符号 `s0`，表示“这个 tile 在全局坐标中的起始行号”；
- 访问 `%tile[%i, %j]` 时，affine map 把逻辑索引 `(d0, d1) = (%i, %j)` 映射成物理索引 ` (d0 + s0, d1) = (%i + row_offset, %j)`。

c++伪代码逻辑：

```c++
float get(float *base_ptr, size_t cols,
          size_t row_offset, size_t i, size_t j) {
    // affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>
    size_t global_row = i + row_offset;  // d0 + s0
    size_t global_col = j;               // d1
    size_t linear_index = global_row * cols + global_col;
    return base_ptr[linear_index];
}
```

## MemRef在编译流程中的位置

在编译流程中，Tensor类型通常是在MemRef类型之上的的类型，其抽象层级较高，Tensor 是纯值语义的，不可变，适合做数学变换。

```C++
%C = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
                   outs(%C_init : tensor<128x512xf32>)
     -> tensor<128x512xf32>

```

这段代码表达的是数学上的矩阵乘法，不涉及具体的内存操作。

经过 tiling、fusion 这些高层优化之后，需要把 Tensor 转成 MemRef，这个过程叫 **bufferization**。

Bufferization 之后，代码变成这样：

```c++
%A_buf = memref.alloc() : memref<128x256xf32>
%B_buf = memref.alloc() : memref<256x512xf32>
%C_buf = memref.alloc() : memref<128x512xf32>
linalg.matmul ins(%A_buf, %B_buf : memref<128x256xf32>, memref<256x512xf32>)
              outs(%C_buf : memref<128x512xf32>)
```

现在每个数组都有了具体的内存分配，可以做 load/store 这些有副作用的操作了。

后续还有一系列 MemRef 相关的优化 pass：

- 把复杂 layout 规范化成简单的 identity layout
- 分析内存的生命周期，自动插入 dealloc
- 把局部的 memref 提升成 SSA 寄存器值
- 根据内存空间生成不同的访问指令

最后降级到 LLVM IR 时，MemRef 会转成一个结构体：

```c++
struct {
    float *allocated_ptr;  // 原始分配的指针
    float *aligned_ptr;    // 对齐后的指针
    size_t offset;         // 偏移量
    size_t shape[2];       // [128, 256]
    size_t stride[2];      // [256, 1]
};
```

**MemRef 和 Tensor 的对比**

| 维度     | Tensor                    | MemRef                         |
| -------- | ------------------------- | ------------------------------ |
| 语义     | 不可变值                  | 可变引用                       |
| 操作     | 纯函数式（无副作用）      | 有副作用（load/store）         |
| 内存     | 抽象的，不涉及具体分配    | 具体的内存 buffer              |
| 优化目标 | 数学变换（fusion/tiling） | 内存访问（locality/bandwidth） |
| 使用阶段 | 编译早期                  | Bufferization 之后             |
| 后端映射 | 不直接降级                | 降级成内存 descriptor          |

**MemRef 是连接高层抽象和底层内存的桥梁**。它在类型系统里保留了足够的信息，让编译器能做各种内存相关的优化，同时又保持了结构化的表达，不像裸指针那样信息全丢了。

## MemRef 的内存分配操作

