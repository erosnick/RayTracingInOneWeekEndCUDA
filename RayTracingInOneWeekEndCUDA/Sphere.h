#pragma once

#include "CUDA.h"
#include "Vec3.h"

struct HitResult {
    bool bHit = false;
    Float t = -Math::infinity;
    Float3 position;
    Float3 normal;
    bool bFrontFace = true;

    inline CUDA_HOST_DEVICE void setFaceNormal(const Ray& ray, const Float3& outwardNormal) {
        bFrontFace = dot(ray.direction, outwardNormal) < Math::epsilon;
        normal = bFrontFace ? outwardNormal : -outwardNormal;
    }
};

class Sphere {
public:
    inline CUDA_HOST_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
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

        hitResult.bHit = true;
        hitResult.t = root;
        hitResult.position = ray.at(hitResult.t);
        auto outwardNormal = (hitResult.position - center) / radius;
        hitResult.setFaceNormal(ray, outwardNormal);
        return true;
    }

    Float3 center;
    Float3 color;
    Float radius;
};