#pragma once

#include <cstdint>
#include <string>
#include <glog/logging.h>

namespace base {
    class NoCopyable {
    protected:
        NoCopyable() = default;

        ~NoCopyable() = default;

        NoCopyable(const NoCopyable &) = delete;

        NoCopyable &operator=(const NoCopyable &) = delete;
    };

    enum class DeviceType : uint8_t {
        kDeviceUnknown = 0,
        kDeviceCPU = 1,
        kDeviceCUDA = 2,
    };

    enum StatusCode : uint8_t {
        kSuccess = 0,
        kFunctionUnImplement = 1,
        kPathNotValid = 2,
        kModelParseError = 3,
        kInternalError = 4,
        kKeyValueHasExist = 5,
        kInvalidArgument = 6,
    };

    enum class DataType : uint8_t {
        kDataTypeUnknown = 0,
        kDataTypeFp32 = 1,
        kDataTypeInt8 = 2,
        kDataTypeInt32 = 3,
    };

    inline size_t DataTypeSize(DataType data_type) {
        if (data_type == DataType::kDataTypeFp32) {
            return sizeof(float);
        } else if (data_type == DataType::kDataTypeInt8) {
            return sizeof(int8_t);
        } else if (data_type == DataType::kDataTypeInt32) {
            return sizeof(int32_t);
        } else {
            LOG(FATAL) << "Invalid data type: " << static_cast<int>(data_type);
            return 0;
        }
    }

    class Status {
    public:
        explicit Status(StatusCode code = StatusCode::kSuccess, std::string err_message = "");

        Status(const Status &other) = default;

        Status &operator=(const Status &other) = default;

        Status &operator=(int code);

        bool operator==(int code) const;

        bool operator!=(int code) const;

        explicit operator int() const;

        explicit operator bool() const;

        int32_t get_err_code() const;

        const std::string &get_err_message() const;

        void set_err_message(const std::string &err_message);

    private:
        int code_ = kSuccess;
        std::string message_;
    };

    namespace error {
        Status FunctionNotImplement(const std::string &err_msg = "");

        Status Success(const std::string &err_msg = "");

        Status PathNotValid(const std::string &err_msg = "");

        Status ModelParseError(const std::string &err_msg = "");

        Status InternalError(const std::string &err_msg = "");

        Status KeyHasExists(const std::string &err_msg = "");

        Status InvalidArgument(const std::string &err_msg = "");
    }

    std::ostream &operator<<(std::ostream &os, const Status &status);
}
