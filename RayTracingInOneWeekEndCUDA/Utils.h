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

        std::cout << "ʹ�õ�GPU device��" << deviceIndex << ": " << devicePro.name << std::endl;
        std::cout << "SM��������" << devicePro.multiProcessorCount << std::endl;
        std::cout << "ÿ���߳̿�Ĺ����ڴ��С��" << devicePro.sharedMemPerBlock / 1024.0 << "KB\n";
        std::cout << "ÿ��SM������߳̿�����" << devicePro.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "ÿ���߳̿������߳�����" << devicePro.maxThreadsPerBlock << std::endl;
        std::cout << "ÿ��SM������߳�����" << devicePro.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "ÿ��SM������߳�������" << devicePro.warpSize << std::endl;
    }
}
//
//namespace Color {
//    inline Vec3 random() {
//        return Vec3(Utils::randomFloat(), Utils::randomFloat(), Utils::randomFloat());;
//    }
//}