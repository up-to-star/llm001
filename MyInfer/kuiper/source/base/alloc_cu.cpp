#include <cuda_runtime_api.h>
#include "base/alloc.h"

namespace base {
    std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

    CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {
    }

    void *CUDADeviceAllocator::allocate(size_t byte_size) const {
        int id = -1;
        cudaError_t state = cudaGetDevice(&id);
        CHECK(state == cudaSuccess);
        if (byte_size > 1024 * 1024) {
            auto &big_buffers = big_buffers_map_[id];
            int sel_id = -1;
            for (size_t i = 0; i < big_buffers.size(); ++i) {
                if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
                    big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) {
                    if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
                        sel_id = i;
                    }
                }
            }
            if (sel_id != -1) {
                big_buffers[sel_id].busy = true;
                return big_buffers[sel_id].data;
            }
            void *ptr = nullptr;
            state = cudaMalloc(&ptr, byte_size);
            if (state != cudaSuccess) {
                char buf[256];
                snprintf(buf, 256,
                        "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory left on device.",
                        byte_size >> 20);
                LOG(ERROR) << buf;
                return nullptr;
            }
            big_buffers.emplace_back(ptr, byte_size, true);
            return ptr;
        }

        auto &cuda_buffers = cuda_buffers_map_[id];
        for (auto & cuda_buffer : cuda_buffers) {
            if (!cuda_buffer.busy && cuda_buffer.byte_size >= byte_size) {
                cuda_buffer.busy = true;
                no_busy_cnt_[id] -= cuda_buffer.byte_size;
                return cuda_buffer.data;
            }
        }
        void *ptr = nullptr;
        state = cudaMalloc(&ptr, byte_size);
        if (state != cudaSuccess) {
            char buf[256];
            snprintf(buf, 256,
                    "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory left on device.",
                    byte_size >> 20);
            LOG(ERROR) << buf;
            return nullptr;
        }
        cuda_buffers.emplace_back(ptr, byte_size, true);
        return ptr;
    }

    void CUDADeviceAllocator::release(void *ptr) const {
        if (!ptr) {
            return;
        }
        if (cuda_buffers_map_.empty()) {
            return;
        }
        cudaError_t state = cudaSuccess;
        for (auto &it : cuda_buffers_map_) {
            if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {
                auto &cuda_buffers = it.second;
                std::vector<CudaMemoryBuffer> temp;
                for (auto &cuda_buffer : cuda_buffers) {
                    if (!cuda_buffer.busy) {
                        state = cudaSetDevice(it.first);
                        state = cudaFree(cuda_buffer.data);
                        CHECK(state == cudaSuccess) << "Error: CUDA error when memory on device " << it.first;
                    } else {
                        temp.emplace_back(cuda_buffer);
                    }
                }
                cuda_buffers.clear();
                it.second = temp;
                no_busy_cnt_[it.first] = 0;
            }
        }

        for (auto &it : cuda_buffers_map_) {
            auto &cuda_buffers = it.second;
            for (auto &cuda_buffer : cuda_buffers) {
                if (cuda_buffer.data == ptr) {
                    no_busy_cnt_[it.first] += cuda_buffer.byte_size;
                    cuda_buffer.busy = false;
                    return;
                }
            }
            auto &big_buffers = big_buffers_map_[it.first];
            for (auto &big_buffer : big_buffers) {
                if (big_buffer.data == ptr) {
                    big_buffer.busy = false;
                    return;
                }
            }
        }
        state = cudaFree(ptr);
        CHECK(state == cudaSuccess) << "Error: CUDA error when free memory on device";
    }

}