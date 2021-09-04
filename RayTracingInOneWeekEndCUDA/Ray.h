#pragma once

#include "CUDATypes.h"

class Ray {
public:
    CUDA_HOST_DEVICE inline Ray() {}
    CUDA_HOST_DEVICE inline Ray(const Float3& inOrigin, const Float3& inDirection)
        : origin(inOrigin), direction(inDirection) {
    }

    CUDA_HOST_DEVICE inline Float3 at(Float t) const {
        return origin + t * direction;
    }

    Float3 origin;
    Float3 direction;
};

