#pragma once

#include "CUDA.h"
#include "Vec3.h"

class Sphere {
public:
    inline CUDA_HOST_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax) const {
        //auto oc = ray.origin - center;
        //auto a = dot(ray.direction, ray.direction);
        //auto b = 2.0f * dot(oc, ray.direction);
        //auto c = dot(oc, oc) - radius * radius;

        //auto discriminant = b * b - 4 * a * c;
        auto oc = ray.origin - center;
        auto a = lengthSquared(ray.direction);
        auto halfB = dot(oc, ray.direction);
        auto c = lengthSquared(oc) - radius * radius;
        auto discriminant = halfB * halfB - a * c;

        auto bHit = (discriminant > Math::epsilon);

        if (!bHit) {
            return false;
        }

        auto d = sqrt(discriminant);

        Float root = (-halfB - d) / a;

        if (root < tMin || tMax < root) {
            root = (-halfB + d) / a;
            if (root < tMin || tMax < root) {
                return false;
            }
        }

        return true;
    }

    Float3 center;
    Float3 color;
    Float radius;
};