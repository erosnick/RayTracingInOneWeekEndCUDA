#pragma once

#include <cstdint>

#include "Vec3.h"
#include "Ray.h"
#include "Constants.h"

class Camera {
public:
    CUDA_HOST_DEVICE Camera(int32_t inImageWidth, int32_t inImageHeight, Float inFOV = 90.0f) {
        initialize(inImageWidth, inImageHeight, inFOV);
    }

    CUDA_HOST_DEVICE void initialize(int32_t inImageWidth, int32_t inImageHeight, Float inFOV = 90.0f) {
        imageWidth = inImageWidth;
        imageHeight = inImageHeight;
        fov = inFOV;

        imageAspectRatio = Float(imageWidth) / imageHeight;
        focalLength = 1.0f;

        scale = tan(Math::radians(fov / 2.0f));

        viewportHeight = 2.0f;
        viewportWidth = viewportHeight * imageAspectRatio;

        horizontal = make_float3(viewportWidth, 0.0f, 0.0f) * scale;
        vertical = make_float3(0.0f, viewportHeight, 0.0f) * scale;
        origin = make_float3(0.0f, 0.0f, 0.0f);
        lowerLeftCorner = origin - horizontal / 2.0f - vertical / 2.0f - make_float3(0.0f, 0.0f, focalLength);
    }

    ~Camera() {
        printf("I'm dead.\n");
    }

    inline CUDA_DEVICE int32_t getImageWidth() const {
        return imageWidth;
    }

    inline CUDA_DEVICE int32_t getImageHeight() const {
        return imageHeight;
    }

    inline CUDA_DEVICE Ray getRay(Float dx, Float dy) {
        auto direction = lowerLeftCorner + dx * horizontal + dy * vertical;
        return Ray(origin, normalize(direction));
    }
private:
    int32_t imageWidth;
    int32_t imageHeight;

    Float imageAspectRatio;
    Float focalLength = 1.0f;
    Float fov;
    Float scale = 1.0f;
    Float viewportHeight;
    Float viewportWidth;

    Float3 horizontal;
    Float3 vertical;
    Float3 origin;
    Float3 lowerLeftCorner;
};