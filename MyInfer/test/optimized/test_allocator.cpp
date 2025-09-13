#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"

TEST(test_buffer, allocate) {
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    Buffer buffer(32, alloc);
    CHECK_NE(buffer.ptr(), nullptr);
}

TEST(test_buffer, use_external) {
    using namespace base;
    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    float *ptr = new float[32];
    Buffer buffer(32, nullptr, ptr, true);
    CHECK_EQ(buffer.is_external(), true);
    CHECK_EQ(buffer.ptr(), ptr);
    delete[] ptr;
}

TEST(test_buffer, allocate1) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    {
        Buffer buffer(32, alloc);
        ASSERT_NE(buffer.ptr(), nullptr);
        LOG(INFO) << "HERE1";
    }
    LOG(INFO) << "HERE2";
}

TEST(test_buffer, allocate2) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    std::shared_ptr<Buffer> buffer;
    { buffer = std::make_shared<Buffer>(32, alloc); }
    LOG(INFO) << "HERE";
    ASSERT_NE(buffer->ptr(), nullptr);
}

TEST(test_buffer, use_external1) {
    using namespace base;
    auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
    float* ptr = new float[32];
    Buffer buffer(32, nullptr, ptr, true);
    CHECK_EQ(buffer.is_external(), true);
    cudaFree(buffer.ptr());
}