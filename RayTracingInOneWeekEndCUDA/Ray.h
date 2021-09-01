#pragma once

class Ray {
public:
    inline CUDA_HOST_DEVICE Ray() {}
    inline CUDA_HOST_DEVICE Ray(const Float3& inOrigin, const Float3& inDirection)
        : origin(inOrigin), direction(inDirection) {
    }

    inline CUDA_HOST_DEVICE Float3 at(Float t) const {
        return origin + t * direction;
    }

    Float3 origin;
    Float3 direction;
};

