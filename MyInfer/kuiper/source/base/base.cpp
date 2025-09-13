#include "base/base.h"

namespace base {
    Status::Status(const StatusCode code, std::string err_message) : code_(code), message_(std::move(err_message)) {
    }

    Status& Status::operator=(const int code) {
        code_ = code;
        return *this;
    }

    bool Status::operator==(const int code) const {
        return code_ == code;
    }

    bool Status::operator!=(const int code) const {
        return code_ != code;
    }

    Status::operator int() const {
        return code_;
    }


    Status::operator bool() const {
        return code_ == kSuccess;
    }

    int32_t Status::get_err_code() const {
        return code_;
    }

    const std::string & Status::get_err_message() const {
        return message_;
    }

    void Status::set_err_message(const std::string &err_message) {
        message_ = err_message;
    }

    namespace error {
        Status Success(const std::string &err_msg) {
            return Status{kSuccess, err_msg};
        }

        Status InvalidArgument(const std::string &err_msg) {
            return Status{kInvalidArgument, err_msg};
        }

        Status FunctionNotImplement(const std::string &err_msg) {
            return Status{kFunctionUnImplement, err_msg};
        }

        Status InternalError(const std::string &err_msg) {
            return Status{kInternalError, err_msg};
        }

        Status KeyHasExists(const std::string &err_msg) {
            return Status{kKeyValueHasExist, err_msg};
        }

        Status ModelParseError(const std::string &err_msg) {
            return Status{kModelParseError, err_msg};
        }

        Status PathNotValid(const std::string &err_msg) {
            return Status{kPathNotValid, err_msg};
        }

    }

    std::ostream &operator<<(std::ostream &os, const Status &status) {
        os << "code: " << status.get_err_code() << " message: " << status.get_err_message();
        return os;
    }
}
