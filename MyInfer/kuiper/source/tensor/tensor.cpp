#include "tensor/tensor.h"

#include <numeric>
#include <glog/logging.h>

namespace tensor {

    template <typename T, typename Tp>
    static size_t reduce_dimension(T begin, T end, Tp init) {
        if (begin >= end) {
            return 0;
        }
        const size_t size = std::accumulate(begin, end, init, std::multiplies<>());
        return size;
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0, const bool need_alloc,
        const std::shared_ptr<base::DeviceAllocator> &alloc, void *ptr) : data_type_(data_type) {
        dims_.emplace_back(dim0);
        size_ = dim0;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            if (ptr != nullptr) {
                CHECK(need_alloc == false)
                    << "The need_alloc is is true when ptr parameter is not a null pointer.";
                init_buffer(alloc, data_type_, need_alloc, ptr);
            }
        }
    }

    Tensor::Tensor(const base::DataType data_type, int32_t dim0, int32_t dim1, const bool need_alloc,
                   const std::shared_ptr<base::DeviceAllocator> &alloc, void *ptr) : data_type_(data_type) {
        dims_.emplace_back(dim0);
        dims_.emplace_back(dim1);
        size_ = dim0 * dim1;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            init_buffer(alloc, data_type, need_alloc, ptr);
        }
    }

    Tensor::Tensor(const base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, const bool need_alloc,
        const std::shared_ptr<base::DeviceAllocator> &alloc, void *ptr) : data_type_(data_type) {
        dims_.emplace_back(dim0);
        dims_.emplace_back(dim1);
        dims_.emplace_back(dim2);
        size_ = dim0 * dim1 * dim2;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            init_buffer(alloc, data_type, need_alloc, ptr);
        }
    }

    Tensor::Tensor(const base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3, const bool need_alloc,
        const std::shared_ptr<base::DeviceAllocator> &alloc, void *ptr) : data_type_(data_type) {
        dims_.emplace_back(dim0);
        dims_.emplace_back(dim1);
        dims_.emplace_back(dim2);
        dims_.emplace_back(dim3);
        size_ = dim0 * dim1 * dim2 * dim3;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            init_buffer(alloc, data_type, need_alloc, ptr);
        }
    }

    Tensor::Tensor(const base::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
        std::shared_ptr<base::DeviceAllocator> alloc, void *ptr) : dims_(std::move(dims)), data_type_(data_type) {
        size_ = reduce_dimension(dims.begin(), dims.end(), 1);
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            init_buffer(alloc, data_type_, need_alloc, ptr);
        }
    }

    void Tensor::to_cpu() {
        CHECK_NE(buffer_, nullptr);
        const base::DeviceType device_type = this->device_type();
        if (device_type == base::DeviceType::kDeviceUnknown) {
            LOG(ERROR) << "The device type of the tensor is unknow.";
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            size_t byte_size = this->byte_size();
            auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
            auto cpu_buffer = std::make_shared<base::Buffer>(byte_size, cpu_alloc);
            cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte_size, base::MemcpyKind::kMemcpyCUDA2CPU);
            this->buffer_ = cpu_buffer;
        } else {
            LOG(INFO) << "The device type of the tensor is already cpu.";
        }
    }

    bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
        if (!allocator) {
            LOG(ERROR) << "The allocator parameter in the allocate function is null pointer!";
            return false;
        }
        size_t byte_size = this->byte_size();
        if (!byte_size) {
            LOG(ERROR) << "The byte_size parameter in the allocate function is eaual to zero!";
            return false;
        }

        if (buffer_ && byte_size <= buffer_->byte_size()) {
            if (!need_realloc) {
                return true;
            }
        }

        buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
        if (!buffer_->ptr()) {
            LOG(ERROR) << "The memory allocated is null pointer";
            return false;
        }
        return true;
    }

}
