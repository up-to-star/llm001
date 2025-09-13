#include "base/alloc.h"
#include <cuda_runtime_api.h>

namespace base {
    void DeviceAllocator::memcpy(const void *src_ptr, void *dst_ptr, size_t byte_size,
                                 MemcpyKind memcpy_kind, void *stream, bool need_sync) const {
        CHECK_NE(src_ptr, nullptr);
        CHECK_NE(dst_ptr, nullptr);

        if (!byte_size) {
            return;
        }

        cudaStream_t stream_ = nullptr;
        if (stream != nullptr) {
            stream_ = static_cast<cudaStream_t>(stream);
        }
        switch (memcpy_kind) {
            case MemcpyKind::kMemcpyCPU2CPU:
                std::memcpy(dst_ptr, src_ptr, byte_size);
                break;
            case MemcpyKind::kMemcpyCPU2CUDA:
                if (stream_ == nullptr) {
                    cudaMemcpy(dst_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
                } else {
                    cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
                }
                break;
            case MemcpyKind::kMemcpyCUDA2CPU:
                if (stream_ == nullptr) {
                    cudaMemcpy(dst_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
                } else {
                    cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
                }
                break;
            case MemcpyKind::kMemcpyCUDA2CUDA:
                if (stream_ == nullptr) {
                    cudaMemcpy(dst_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
                } else {
                    cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
                }
                break;
            default:
                LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
        }
        if (need_sync) {
            cudaDeviceSynchronize();
        }
    }

    void DeviceAllocator::memset_zero(void *ptr, size_t byte_size, void *stream, bool need_sync) {
        CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
        if (device_type_ == base::DeviceType::kDeviceCPU) {
            std::memset(ptr, 0, byte_size);
        } else {
            cudaMemset(ptr, 0, byte_size);
        }

        if (need_sync) {
            cudaDeviceSynchronize();
        }
    }


}


