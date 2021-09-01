
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.h"
#include "Canvas.h"
#include <cstdio>

template<typename T>
T* createObject() {
    T* object = nullptr;
    gpuErrorCheck(cudaMallocManaged(&object, sizeof(T*)));
    return object;
}

template<typename T>
void deleteObject(T* object) {
    gpuErrorCheck(cudaFree(object));
}

CUDA_GLOBAL void kernel(Canvas* canvas) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;
    auto width = canvas->getWidth();
    auto height = canvas->getHeight();

    auto index = y * width + x;

    if (index < (width * height)) {
        canvas->writePixel(index, make_float3(Float(x) / width, Float(y) / height, 0.0f));
    }
}

int main() {
    constexpr auto width = 1280;
    constexpr auto height = 720;

    auto canvas = createObject<Canvas>();
    canvas->initialize(width, height);

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    kernel<<<gridSize, blockSize>>>(canvas);
    gpuErrorCheck(cudaDeviceSynchronize());

    canvas->writeToPNG("render.png");
    Utils::openImage(L"render.png");

    canvas->uninitialize();
    deleteObject(canvas);

    return 0;
}
