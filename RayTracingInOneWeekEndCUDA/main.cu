
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.h"
#include "Canvas.h"
#include "GPUTimer.h"
#include "Camera.h"
#include "Sphere.h"
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

constexpr auto SPHERES = 1;
CUDA_CONSTANT Sphere constantSpheres[SPHERES];

CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult) {
    HitResult tempHitResult;
    bool bHitAnything = false;
    Float closestSoFar = tMax;
    for (auto& sphere : constantSpheres) {
        if (sphere.hit(ray, tMin, closestSoFar, tempHitResult)) {
            bHitAnything = true;
            closestSoFar = tempHitResult.t;
            hitResult = tempHitResult;
        }
    }

    return bHitAnything;
}

CUDA_DEVICE Float3 rayColor(const Ray& ray) {
    HitResult hitResult;
    if (hit(ray, Math::epsilon, Math::infinity, hitResult)) {
        //return 0.5f * (hitResult.normal + 1.0f);
        return make_float3(1.0, 0.0, 0.0);
    }

    auto unitDirection = normalize(ray.direction);
    auto t = 0.5f * (unitDirection.y + 1.0f);
    return lerp(make_float3(1.0f, 1.0f, 1.0f), make_float3(0.5f, 0.7f, 1.0f), t);
}

CUDA_GLOBAL void kernel(Canvas canvas, Camera camera) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;
    auto width = camera.getImageWidth();
    auto height = camera.getImageHeight();

    auto index = y * width + x;

    if (index < (width * height)) {
        auto dx = Float(x) / (width - 1);
        auto dy = Float(y) / (height - 1);

        auto ray = camera.getRay(dx, dy);

        auto color = rayColor(ray);

        canvas.writePixel(index, color);
    }
}

int main() {
    //gpuErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    constexpr auto width = 1280;
    constexpr auto height = 720;

    Canvas canvas(width, height);
    //auto* canvas = createObject<Canvas>();
    //canvas->initialize(width, height);

    Camera camera(width, height);
    //auto* camera = createObject<Camera>();
    //camera->initialize(width, height);

    Sphere spheres[SPHERES];

    spheres[0].center = {0.0f, 0.0f, -1.0f};
    spheres[0].color = make_float3(1.0f, 0.0f, 0.0f);
    spheres[0].radius = 0.5f;

    //spheres[1].center = { 0.0f, -100.5f, -1.0f };
    //spheres[1].color = make_float3(1.0f, 0.0f, 0.0f);
    //spheres[1].radius = 100.0f;

    gpuErrorCheck(cudaMemcpyToSymbol(constantSpheres, spheres, sizeof(Sphere) * SPHERES));

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    GPUTimer timer("Rendering start...");

    kernel<<<gridSize, blockSize>>>(canvas, camera);
    gpuErrorCheck(cudaDeviceSynchronize());

    timer.stop("Rendering elapsed time");

    canvas.writeToPNG("render.png");
    Utils::openImage(L"render.png");

    //deleteObject(camera);

    canvas.uninitialize();
    //deleteObject(canvas);

    return 0;
}
