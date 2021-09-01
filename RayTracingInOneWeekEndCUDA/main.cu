
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.h"
#include "Canvas.h"
#include "GPUTimer.h"
#include "Camera.h"
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

CUDA_DEVICE Vec3 rayColor(const Ray& ray) {
    auto unitDirection = normalize(ray.direction);
    auto t = 0.5f * (unitDirection.y + 1.0f);
    return lerp(Color::White(), Color::LightCornflower(), t);
}

CUDA_GLOBAL void kernel(Canvas* canvas, Camera* camera) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;
    auto width = camera->getImageWidth();
    auto height = camera->getImageHeight();

    auto index = y * width + x;

    if (index < (width * height)) {
        auto dx = Float(x) / (width - 1);
        auto dy = Float(y) / (height - 1);

        auto ray = camera->getRay(dx, dy);

        auto color = rayColor(ray);

        canvas->writePixel(index, color);
    }
}

int main() {
    constexpr auto width = 1280;
    constexpr auto height = 720;

    auto* canvas = createObject<Canvas>();
    canvas->initialize(width, height);

    auto* camera = createObject<Camera>();
    camera->initialize(width, height);

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    GPUTimer timer("Rendering start...");

    kernel<<<gridSize, blockSize>>>(canvas, camera);
    gpuErrorCheck(cudaDeviceSynchronize());

    timer.stop("Rendering elapsed time");

    canvas->writeToPNG("render.png");
    Utils::openImage(L"render.png");

    deleteObject(camera);

    canvas->uninitialize();
    deleteObject(canvas);

    return 0;
}
