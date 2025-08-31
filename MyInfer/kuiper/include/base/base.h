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
}