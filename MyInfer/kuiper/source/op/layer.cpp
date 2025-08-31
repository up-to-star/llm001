#include "op/layer.h"


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

    std::string &BaseLayer::get_layer_name() const {
        return layer_name_;
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
}
