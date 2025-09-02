#include "tensor/tensor.h"

#include <numeric>
#include <glog/logging.h>

namespace tensor {
    template<typename T, typename Tp>
    static size_t reduce_dimension(T begin, T end, Tp init) {
        if (begin >= end) {
            return 0;
        }
        const size_t size = std::accumulate(begin, end, init, std::multiplies<>());
        return size;
    }

    Tensor::Tensor(const base::DataType data_type, int32_t dim0, const bool need_alloc,
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

    Tensor::Tensor(const base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                   const bool need_alloc,
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
                   std::shared_ptr<base::DeviceAllocator> alloc, void *ptr) : dims_(std::move(dims)),
                                                                              data_type_(data_type) {
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

    void Tensor::to_cuda(cudaStream_t stream) {
        CHECK_NE(buffer_, nullptr);
        const base::DeviceType device_type = this->device_type();
        if (device_type == base::DeviceType::kDeviceUnknown) {
            LOG(ERROR) << "The device type of the tensor is unknow.";
        } else if (device_type == base::DeviceType::kDeviceCPU) {
            size_t byte_size = this->byte_size();
            auto cu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
            auto cu_buffer = std::make_shared<base::Buffer>(byte_size, cu_alloc);
            cu_alloc->memcpy(cu_buffer->ptr(), buffer_->ptr(), byte_size, base::MemcpyKind::kMemcpyCPU2CUDA, stream);
            this->buffer_ = cu_buffer;
        } else {
            LOG(INFO) << "The device type of the tensor is already cuda.";
        }
    }

    bool Tensor::is_empty() const {
        return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
    }

    static size_t data_type_size(base::DataType data_type) {
        switch (data_type) {
            case base::DataType::kDataTypeFp32:
                return sizeof(float);
            case base::DataType::kDataTypeInt32:
                return sizeof(int32_t);
            case base::DataType::kDataTypeInt8:
                return sizeof(int8_t);
            default:
                LOG(FATAL) << "Unknown data type size for " << static_cast<int>(data_type);
                return 0;
        }
    }

    void Tensor::init_buffer(const std::shared_ptr<base::DeviceAllocator> &alloc, base::DataType data_type,
                             const bool need_alloc,
                             void *ptr) {
        if (!alloc && !need_alloc) {
            auto buffer = std::make_shared<base::Buffer>(data_type_size(data_type) * size_, nullptr, ptr, true);
        } else {
            allocate(alloc, true);
        }
    }

    void Tensor::reshape(const std::vector<int32_t> &dims) {
        const size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
        if  (!buffer_) {
            this->dims_ = dims;
            this->size_ = size;
            return;
        }
        if (size > size_) {
            auto new_buffer = std::make_shared<base::Buffer>(size *  data_type_size(data_type_), buffer_->allocator());
            CHECK(new_buffer->allocate());
            new_buffer->copy_from(buffer_.get());
            buffer_ = new_buffer;
        }
        this->dims_ = dims;
        this->size_ = size;
    }

    std::shared_ptr<base::Buffer> Tensor::get_buffer() const {
        return buffer_;
    }

    size_t Tensor::size() const {
        return size_;
    }

    size_t Tensor::byte_size() const {
        return size_ * data_type_size(data_type_);
    }

    int32_t Tensor::dim_size() const {
        return static_cast<int32_t>(dims_.size());
    }

    base::DataType Tensor::data_type() const {
        return data_type_;
    }

    int32_t Tensor::get_dim(const int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, dims_.size());
        return dims_.at(idx);
    }

    const std::vector<int32_t> & Tensor::dims() const {
        return dims_;
    }

    std::vector<int32_t> Tensor::strides() const {
        std::vector<int32_t> strides;
        if (!dims_.empty()) {
            for (int32_t i = 0; i < dims_.size(); i++) {
                size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
                strides.emplace_back(stride);
            }
            strides.emplace_back(1);
        }
        return strides;
    }

    bool Tensor::assign(const std::shared_ptr<base::Buffer> &buffer) {
        if (!buffer) {
            LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
            return false;
        }
        if (buffer_) {
            if (buffer_->device_type() != buffer->device_type()) {
                LOG(ERROR) << "The device type of the tensor is not equal to the device type of the buffer!";
                return false;
            }
        }
        const size_t byte_size = this->byte_size();
        if (byte_size > buffer->byte_size()) {
            LOG(ERROR) << "The size of buffer is too small for the tensor";
            return false;
        }
        buffer_ = buffer;
        return true;
    }

    void Tensor::reset(base::DataType data_type, const std::vector<int32_t> &dims) {
        this->data_type_ = data_type;
        this->dims_ = dims;
        this->size_ = reduce_dimension(dims.begin(), dims.end(), 1);
        buffer_ = nullptr;
    }

    void Tensor::set_device_type(base::DeviceType device_type) const {
        if (buffer_) {
            buffer_->set_device_type(device_type);
        }
    }

    base::DeviceType Tensor::device_type() const {
        if (!buffer_) {
            return base::DeviceType::kDeviceUnknown;
        }
        return buffer_->device_type();
    }

    bool Tensor::allocate(const std::shared_ptr<base::DeviceAllocator> &allocator, const bool need_realloc) {
        if (!allocator) {
            LOG(ERROR) << "The allocator parameter in the allocate function is null pointer!";
            return false;
        }
        size_t byte_size = this->byte_size();
        if (!byte_size) {
            LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
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

    Tensor Tensor::clone() const {
        Tensor new_tensor = *this;
        size_t byte_size = this->byte_size();
        auto allocator = buffer_->allocator();
        new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
        new_tensor.buffer_->copy_from(buffer_.get());
        return new_tensor;
    }
}
