#pragma once

#include "CUDA.h"
#include "Ray.h"
#include "HitResult.h"
#include "Utils.h"

class Material {
public:
    CUDA_DEVICE inline virtual bool scatter(const Ray& inRay, const HitResult& hitResult, Float3& attenuation, Ray& scattered, curandState* randState) const = 0;
};

class Lambertian : public Material {
public:
    CUDA_DEVICE Lambertian(const Float3& inAlbedo, Float inAbsorb)
        : albedo(inAlbedo), absorb(inAbsorb) {
    }

    CUDA_DEVICE inline bool scatter(const Ray& inRay, const HitResult& hitResult, Float3& attenuation, Ray& scattered, curandState* randState) const override {
        //auto scatterDirection = Utils::randomUnitVector(randState);                                         // Diffuse1
        auto scatterDirection = hitResult.normal + Utils::randomUnitVector(randState);                      // Diffuse2
        //auto scatterDirection = Utils::randomHemiSphere(hitResult.normal, randState);                       // Diffuse3
        //auto scatterDirection = hitResult.normal + Utils::randomHemiSphere(hitResult.normal, randState);    // Diffuse4
        //auto scatterDirection = hitResult.normal + Utils::randomInUnitSphere(randState);                    // Diffuse5
        // Catch degenerate scatter direction
        // If the random unit vector we generate is exactly opposite the normal vector, 
        // the two will sum to zero, which will result in a zero scatter direction vector. 
        // This leads to bad scenarios later on (infinities and NaNs),
        if (Utils::nearZero(scatterDirection)) {
            scatterDirection = hitResult.normal;
        }
        scattered = Ray(hitResult.position, normalize(scatterDirection));
        attenuation = albedo;
        return true;
    }
    Float3 albedo;
    Float absorb;
};

class Metal : public Material {
public:
    CUDA_DEVICE Metal(const Float3& inAlbedo, Float inFuzz = 1.0f)
    : albedo(inAlbedo), fuzz(inFuzz < 1.0f ? inFuzz : 1.0f) {
    }

    CUDA_DEVICE inline bool scatter(const Ray& inRay, const HitResult& hitResult, Float3& attenuation, Ray& scattered, curandState* randState) const override {
        auto reflected = Utils::reflect(normalize(inRay.direction), hitResult.normal);
        scattered = Ray(hitResult.position, reflected + fuzz * Utils::randomInUnitSphere(randState));
        attenuation = albedo;
        return (dot(scattered.direction, hitResult.normal) > 0);
    }

    Float3 albedo;
    Float fuzz;
};