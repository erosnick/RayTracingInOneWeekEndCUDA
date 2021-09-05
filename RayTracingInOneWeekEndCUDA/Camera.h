#pragma once

#include <cstdint>

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
        cameraSpeed = 6.0f;

        scale = tan(Math::radians(fov / 2.0f));

        viewportHeight = 2.0f * scale;
        viewportWidth = viewportHeight * aspectRatio;

        bIsDirty = true;
        updateViewMatrix();
    }

    ~Camera() {
        printf("I'm dead.\n");
    }

    void walk(Float delta) {
        eye += forward * cameraSpeed * delta;
        center += forward * cameraSpeed * delta;
        bIsDirty = true;
    }

    void strafe(Float delta) {
        eye += right * cameraSpeed * delta;
        center += right * cameraSpeed * delta;
        bIsDirty = true;
    }

    void raise(Float delta) {
        eye += up * cameraSpeed * delta;
        center += up * cameraSpeed * delta;
        bIsDirty = true;
    }

    CUDA_HOST_DEVICE void updateViewMatrix() {
        if (bIsDirty) {
            forward = normalize(eye - center);
            right = normalize(cross(up, forward));
            up = cross(forward, right);

            origin = eye;
            horizontal = viewportWidth * right;
            vertical = viewportHeight * up;
            lowerLeftCorner = origin - horizontal / 2.0f - vertical / 2.0f - forward;
        }
    }

    bool isDirty() const {
        return bIsDirty;
    }

    void resetDiryFlag() {
        bIsDirty = false;
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
    Float cameraSpeed = 6.0f;

    Float3 eye;
    Float3 center;
    Float3 forward;
    Float3 right;
    Float3 up;
    Float3 horizontal;
    Float3 vertical;
    Float3 origin;
    Float3 lowerLeftCorner;

    bool bIsDirty;
};