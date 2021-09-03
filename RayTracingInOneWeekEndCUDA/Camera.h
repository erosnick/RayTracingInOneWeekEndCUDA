#pragma once

#include <cstdint>

#include "Vec3.h"
#include "Ray.h"
#include "Constants.h"

class Camera {
public:
    CUDA_HOST_DEVICE Camera(const Float3& inEye, const Float3& inCenter, const Float3& inUp, Float inAspectRatio, Float inFOV = 90.0f) {
        initialize(inEye, inCenter, inUp, inAspectRatio, inFOV);
    }

    CUDA_HOST_DEVICE void initialize(const Float3& inEye, const Float3& inCenter, const Float3& inUp, Float inAspectRatio, Float inFOV = 90.0f) {
        eye = inEye;
        center = inCenter;
        up = inUp;
        aspectRatio = inAspectRatio;
        fov = inFOV;

        focalLength = 1.0f;

        scale = tan(Math::radians(fov / 2.0f));

        viewportHeight = 2.0f * scale;
        viewportWidth = viewportHeight * aspectRatio;

        auto w = normalize(eye - center);
        auto u = normalize(cross(up, w));
        auto v = cross(w, u);

        origin = eye;
        horizontal = viewportWidth * u;
        vertical = viewportHeight * v;
        lowerLeftCorner = origin - horizontal / 2.0f - vertical / 2.0f - w;
    }

    ~Camera() {
        printf("I'm dead.\n");
    }

    CUDA_DEVICE inline Ray getRay(Float dx, Float dy) {
        auto direction = lowerLeftCorner + dx * horizontal + dy * vertical - origin;
        return Ray(origin, normalize(direction));
    }
private:
    Float aspectRatio;
    Float focalLength = 1.0f;
    Float fov;
    Float scale = 1.0f;
    Float viewportHeight;
    Float viewportWidth;
    Float3 eye;
    Float3 center;
    Float3 up;
    Float3 horizontal;
    Float3 vertical;
    Float3 origin;
    Float3 lowerLeftCorner;
};