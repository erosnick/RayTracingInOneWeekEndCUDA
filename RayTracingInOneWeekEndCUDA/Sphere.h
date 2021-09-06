#pragma once

#include "CUDA.h"
#include "Vec3.h"
#include "HitResult.h"
#include "Material.h"

class Sphere {
public:
    void initialize(const Float3& inCenter, float inRadius, Material* inMaterial) {
        center = inCenter;
        radius = inRadius;
        material = inMaterial;
    }

    void uninitailize() {
        //if (material) {
        //    delete material;
        //}
    }

    CUDA_DEVICE inline bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) const {
        //auto oc = ray.origin - center;
        //auto a = dot(ray.direction, ray.direction);
        //auto b = 2.0f * dot(oc, ray.direction);
        //auto c = dot(oc, oc) - radius * radius;

        //auto discriminant = b * b - 4 * a * c;
        auto oc = ray.origin - center;
        auto a = dot(ray.direction, ray.direction);
        auto halfB = dot(oc, ray.direction);
        auto c = dot(oc, oc) - radius * radius;
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
        hitResult.material = material;
        return true;
    }

    Float3 center;
    Float radius;
    Material* material;
};