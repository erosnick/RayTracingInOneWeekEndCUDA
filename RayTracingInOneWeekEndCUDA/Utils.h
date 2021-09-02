#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include "CUDA.h"

#include <string>
#include <algorithm>
#include <random>
#include <iostream>

#define gpuErrorCheck(ans) { Utils::gpuAssert((ans), __FILE__, __LINE__); }

namespace Utils {
    inline float randomFloat(float start = 0.0f, float end = 1.0f) {
        std::uniform_real_distribution<float> distribution(start, end);
        static std::random_device randomDevice;
        static std::mt19937 generator(randomDevice());
        return distribution(generator);
    }

    inline double randomDouble(double start = 0.0, double end = 1.0) {
        std::uniform_real_distribution<double> distribution(start, end);
        static std::random_device randomDevice;
        static std::mt19937 generator(randomDevice());
        return distribution(generator);
    }

    // Random double in [0, 1]
    inline CUDA_DEVICE Float random(curandState* randState) {
        return curand_uniform(randState);
    }

    // Random float3 in [0, 1]
    inline CUDA_DEVICE Float3 randomVector(curandState* randState) {
        Float x = curand_uniform(randState);
        Float y = curand_uniform(randState);
        Float z = curand_uniform(randState);

        return make_float3(x, y, z);
    }

    // Random float3 in [min, max]
    inline CUDA_DEVICE Float3 randomVector(curandState* randState, Float min, Float max) {
        return (min + (max - min) * randomVector(randState));
    }

    inline CUDA_DEVICE Float3 randomInUnitSphere(curandState* randState) {
        while (true) {
            auto position = randomVector(randState, -1.0f, 1.0f);
            if (lengthSquared(position) >= 1.0f) {
                continue;
            }

            return position;
        }
    }

    inline CUDA_DEVICE Float3 randomUnitVector(curandState* randState) {
        return normalize(randomInUnitSphere(randState));
    }

    inline CUDA_DEVICE Float3 randomHemiSphere(const Float3& normal, curandState* randState) {
        auto inUnitSphere = randomInUnitSphere(randState);

        //  In the same hemisphere as the normal
        if (dot(inUnitSphere, normal) > 0.0f) {
            return inUnitSphere;
        }
        return -inUnitSphere;
    }

    void openImage(const std::wstring& path);

    inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
        if (code != cudaSuccess)
        {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            // Make sure we call CUDA Device Reset before exiting
            cudaDeviceReset();
            if (abort) exit(code);
        }
    }

    inline void reportGPUUsageInfo() {
        size_t freeBytes;
        size_t totalBytes;

        gpuErrorCheck(cudaMemGetInfo(&freeBytes, &totalBytes));

        auto freeDb = (double)freeBytes;

        auto totalDb = (double)totalBytes;

        auto usedDb = totalDb - freeDb;

        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
            usedDb / 1024.0 / 1024.0, freeDb / 1024.0 / 1024.0, totalDb / 1024.0 / 1024.0);
    }

    inline void queryDeviceProperties() {
        int32_t deviceIndex = 0;
        cudaDeviceProp devicePro;
        cudaGetDeviceProperties(&devicePro, deviceIndex);

        std::cout << "使用的GPU device：" << deviceIndex << ": " << devicePro.name << std::endl;
        std::cout << "SM的数量：" << devicePro.multiProcessorCount << std::endl;
        std::cout << "每个线程块的共享内存大小：" << devicePro.sharedMemPerBlock / 1024.0 << "KB\n";
        std::cout << "每个SM的最大线程块数：" << devicePro.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "每个线程块的最大线程数：" << devicePro.maxThreadsPerBlock << std::endl;
        std::cout << "每个SM的最大线程数：" << devicePro.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个SM的最大线程束数：" << devicePro.warpSize << std::endl;
    }
}
//
//namespace Color {
//    inline Vec3 random() {
//        return Vec3(Utils::randomFloat(), Utils::randomFloat(), Utils::randomFloat());;
//    }
//}