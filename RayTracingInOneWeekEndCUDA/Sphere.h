#pragma once

#include "CUDATypes.h"
#include "Vec3.h"
#include "HitResult.h"
#include "Material.h"

class Sphere {
public:
    void initialize(const Float3& inCenter, float inRadius, Material* inMaterial) {
        center = inCenter;
        radius = inRadius;
        material = inMaterial;
        bShading = true;
    }

    void uninitailize() {
        //if (material) {
        //    delete material;
        //}
    }

    CUDA_DEVICE inline bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
        auto oc = ray.origin - center;
        auto a = dot(ray.direction, ray.direction);
        auto halfB = dot(oc, ray.direction);
        auto c = dot(oc, oc) - radius * radius;
        auto discriminant = halfB * halfB - a * c;
        // Cant's use Math::epsilon(0.001f) for comparison here
        // Because it's not small enough(Not precise enough)
        auto bHit = (discriminant > 0.0f);

        if (!bHit) {
            return false;
        }

        auto sqrtd = sqrt(discriminant);
        Float root = (-halfB - sqrtd) / a;

        // Find the nearest root that lies in the acceptable range.
        if (root < tMin || tMax < root) {
            root = (-halfB + sqrtd) / a;
            if (root < tMin || tMax < root) {
                return false;
            }
        }

        hitResult.bHit = true;
        hitResult.t = root;
        hitResult.position = ray.at(hitResult.t);
        auto outwardNormal = (hitResult.position - center) / radius;
        hitResult.setFaceNormal(ray, outwardNormal);
        hitResult.material = material;
        return true;
    }

    Float3 center;
    Float radius;
    Material* material;
    bool bShading;
};