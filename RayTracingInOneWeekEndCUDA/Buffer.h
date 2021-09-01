#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDA.h"
#include "Utils.h"

#include <cstdint>
#include <cassert>

class Buffer {
public:
    Buffer() {}
    Buffer(int32_t inSize) {
        initialize(inSize);
    }

    ~Buffer() {
        uninitialize();
    }

    void initialize(int32_t inSize) {
        size = inSize;
        gpuErrorCheck(cudaMallocManaged(&buffer, size));
    }

    void uninitialize() {
        gpuErrorCheck(cudaFree(buffer));
    }

    CUDA_HOST_DEVICE uint8_t operator[](int32_t index) const {
        return buffer[index];
    }

    CUDA_HOST_DEVICE uint8_t& operator[](int32_t index) {
        if (index >= size)
        {
            printf("Index out of range.\n");
            assert(index < size);
        }
        return buffer[index];
    }

    uint8_t* get() {
        return buffer;
    }

    uint8_t* buffer = nullptr;
    int32_t size = 0;
};