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
    inline float randomFloat(double start = 0.0f, double end = 1.0f) {
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

    // random number generator from https://github.com/gz/rust-raytracer

    inline CUDA_DEVICE float getrandom(uint32_t* seed0, uint32_t* seed1) {
        *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
        *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

        uint32_t ires = ((*seed0) << 16) + (*seed1);

        // Convert to Float
        union {
            float f;
            uint32_t ui;
        } res;

        res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

        return (res.f - 2.0f) / 2.0f;
    }

    //// Random double in [0, 1]
    //inline CUDA_DEVICE Float random(unsigned long long seed) {
    //    curandState state;
    //    curand_init((unsigned long long)clock() + seed, 0, 0, &state);

    //    return curand_uniform_double(&state);
    //}

    //// Random float3 in [0, 1]
    //inline CUDA_DEVICE Float3 randomVector(unsigned long long seed) {
    //    curandState state;
    //    curand_init((unsigned long long)clock() + seed, 0, 0, &state);

    //    Float x = curand_uniform_double(&state);
    //    Float y = curand_uniform_double(&state);
    //    Float z = curand_uniform_double(&state);

    //    return make_float3(x, y, z);
    //}

    //// Random float3 in [min, max]
    //inline CUDA_DEVICE Float3 randomVector(unsigned long long seed, Float min, Float max) {
    //    curandState state;
    //    curand_init((unsigned long long)clock() + seed, 0, 0, &state);

    //    Float x = min + (max - min) * curand_uniform_double(&state);
    //    Float y = min + (max - min) * curand_uniform_double(&state);
    //    Float z = min + (max - min) * curand_uniform_double(&state);

    //    return make_float3(x, y, z);
    //}

    //inline CUDA_DEVICE Float3 randomInUnitSphere(unsigned long long seed) {
    //    while (true) {
    //        auto position = randomVector(seed, -1.0f, 1.0f);
    //        if (lengthSquared(position) >= 1.0f) {
    //            continue;
    //        }

    //        return position;
    //    }
    //}

    // Random float3 in [0, 1]
    inline CUDA_DEVICE Float3 randomVector(uint32_t* seed0, uint32_t* seed1) {
        Float x = getrandom(seed0, seed1);
        Float y = getrandom(seed0, seed1);
        Float z = getrandom(seed0, seed1);
        return make_float3(x, y, z);
    }

    // Random float3 in [min, max]
    inline CUDA_DEVICE Float3 randomVector(uint32_t* seed0, uint32_t* seed1, Float min, Float max) {
        return min + (max - min) * randomVector(seed0, seed1);
    }

    inline CUDA_DEVICE Float3 randomInUnitSphere(uint32_t* seed0, uint32_t* seed1) {
        while (true) {
            auto position = randomVector(seed0, seed1);
            //position = 2.0f * position - 1.0f;
            if (lengthSquared(position) >= 1.0f) {
                //printf("shit!!!\n");
                continue;
            }
            
            return position;
        }
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