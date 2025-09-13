#include "op/layer.h"

#include <cstdarg>
#include <utility>


namespace op {
    BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                         std::string layer_name)
        : layer_name_(std::move(layer_name)), layer_type_(layer_type), data_type_(data_type),
          device_type_(device_type) {
    }

    base::DataType BaseLayer::data_type() const {
        return data_type_;
    }

    LayerType BaseLayer::layer_type() const {
        return layer_type_;
    }

    std::string BaseLayer::get_layer_name() const {
        return layer_name_;
    }

    base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor &weight) {
        return base::error::FunctionNotImplement();
    }

    base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t> &dims, const void *weight_ptr,
                                       base::DeviceType device_type) {
        return base::error::FunctionNotImplement();
    }

    void BaseLayer::set_layer_name(const std::string &layer_name) {
        layer_name_ = layer_name;
    }

    base::DeviceType BaseLayer::device_type() const {
        return device_type_;
    }

    void BaseLayer::set_device_type(base::DeviceType device_type) {
        device_type_ = device_type;
    }

    Layer::Layer(const base::DeviceType device_type, const LayerType layer_type,
                 std::string layer_name) : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32,
                                                     std::move(layer_name)) {
    }

    base::Status Layer::init() {
        return base::error::Success();
    }

    base::Status Layer::forward() {
        return base::error::FunctionNotImplement();
    }

    base::Status Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &output1) {
        set_input(0, input1);
        set_output(0, output1);
        return forward();
    }

    base::Status Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
        const tensor::Tensor &output1) {
        set_input(0, input1);
        set_input(1, input2);
        set_output(0, output1);
        return forward();
    }

    base::Status Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
        const tensor::Tensor &input3, const tensor::Tensor &output1) {
        set_input(0, input1);
        set_input(1, input2);
        set_input(2, input3);
        set_output(0, output1);
        return forward();
    }

    base::Status Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
        const tensor::Tensor &input3, const tensor::Tensor &input4, const tensor::Tensor &output1) {
        set_input(0, input1);
        set_input(1, input2);
        set_input(2, input3);
        set_input(3, input4);
        set_output(0, output1);
        return forward();
    }

    base::Status Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
        const tensor::Tensor &input3, const tensor::Tensor &input4, const tensor::Tensor &input5,
        const tensor::Tensor &output1) {
        set_input(0, input1);
        set_input(1, input2);
        set_input(2, input3);
        set_input(3, input4);
        set_input(4, input5);
        set_output(0, output1);
        return forward();
    }

    void Layer::set_input(int32_t idx, const tensor::Tensor &input) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        inputs_.at(idx) = input;
    }

    void Layer::set_output(const int32_t idx, const tensor::Tensor &output) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        outputs_.at(idx) = output;
    }

    tensor::Tensor &Layer::get_input(const int32_t idx) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        return inputs_.at(idx);
    }

    const tensor::Tensor &Layer::get_input(const int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        return inputs_.at(idx);
    }

    const tensor::Tensor &Layer::get_output(const int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        return outputs_.at(idx);
    }

    tensor::Tensor &Layer::get_output(const int32_t idx) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        return outputs_.at(idx);
    }

    size_t Layer::input_size() const {
        return inputs_.size();
    }

    size_t Layer::output_size() const {
        return outputs_.size();
    }

    void Layer::to_cuda() {
        for (auto &input : inputs_) {
            if (!input.is_empty()) {
                input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
            }
        }

        for (auto &output : outputs_) {
            if (!output.is_empty()) {
                output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
            }
        }
    }

    void Layer::reset_input_size(size_t size) {
        inputs_.resize(size);
    }

    void Layer::reset_output_size(size_t size) {
        outputs_.resize(size);
    }

    void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
        if (!config) {
            return;
        }
        cuda_config_ = std::move(config);
    }

    std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const {
        return cuda_config_;
    }


    base::Status Layer::check() const {
        return base::error::FunctionNotImplement("The check function is not implement yet.");
    }

    base::Status Layer::check_tensor(const tensor::Tensor &tensor, const base::DeviceType device_type,
                                     const base::DataType data_type) const {
        if (tensor.is_empty()) {
            return base::error::InvalidArgument("The tensor parameter is empty.");
        }
        if (tensor.device_type() != device_type) {
            return base::error::InvalidArgument("The tensor device type is not equal to the layer device type.");
        }
        if (tensor.data_type() != data_type) {
            return base::error::InvalidArgument("The tensor data type is not equal to the layer data type.");
        }
        return base::error::Success();
    }

    base::Status Layer::check_tensor_with_dim(const tensor::Tensor &tensor, base::DeviceType device_type,
                                              base::DataType data_type, ...) const {
        std::va_list args;
        if (tensor.is_empty()) {
            return base::error::InvalidArgument("The tensor parameter is empty.");
        }
        if (tensor.device_type() != device_type) {
            return base::error::InvalidArgument("The tensor device type is not equal to the layer device type.");
        }
        if (tensor.data_type() != data_type) {
            return base::error::InvalidArgument("The tensor data type is not equal to the layer data type.");
        }
        va_start(args, data_type);
        const int32_t dims = tensor.dim_size();
        for (int32_t i = 0; i < dims; i++) {
            const int32_t dim = va_arg(args, int32_t);
            if (dim != tensor.get_dim(i)) {
                return base::error::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
            }
        }
        va_end(args);
        return base::error::Success();
    }
}
