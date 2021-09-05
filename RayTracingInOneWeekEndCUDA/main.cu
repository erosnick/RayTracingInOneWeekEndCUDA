
#include "main.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include "Utils.h"
#include "GPUTimer.h"
#include "Sphere.h"
#include <cstdio>

template<typename T>
T* createObjectPtr() {
    T* object = nullptr;
    gpuErrorCheck(cudaMallocManaged(&object, sizeof(T*)));
    return object;
}

template<typename T>
T* createObjectArray(int32_t numObjects) {
    T* object = nullptr;
    gpuErrorCheck(cudaMallocManaged(&object, sizeof(T) * numObjects));
    return object;
}

template<typename T>
T* createObjectPtrArray(int32_t numObjects) {
    T* object = nullptr;
    gpuErrorCheck(cudaMallocManaged(&object, sizeof(T*) * numObjects));
    return object;
}

template<typename T>
void deleteObject(T* object) {
    gpuErrorCheck(cudaFree(object));
}

template<typename T>
CUDA_GLOBAL void deleteDeviceObject(T** object) {
    delete (*object);
}

constexpr auto SPHERES = 5;
CUDA_CONSTANT Sphere constantSpheres[SPHERES];

CUDA_DEVICE bool hit(const Ray& ray, Float tMin, Float tMax, HitResult& hitResult, Sphere* spheres) {
    HitResult tempHitResult;
    bool bHitAnything = false;
    Float closestSoFar = tMax;
    //for (auto& sphere : constantSpheres) {
    for (auto i = 0; i < SPHERES; i++){
        auto sphere = spheres[i];
        if (!sphere.bShading) {
            continue;
        }
        if (sphere.hit(ray, tMin, closestSoFar, tempHitResult)) {
            bHitAnything = true;
            closestSoFar = tempHitResult.t;
            hitResult = tempHitResult;
        }
    }

    return bHitAnything;
}

CUDA_DEVICE Float3 rayColor(const Ray& ray, curandState* randState, Sphere* spheres) {
    Ray currentRay = ray;
    auto currentAttenuation = make_float3(1.0f, 1.0f, 1.0f);
    for (auto i = 0; i < 50; i++) {
        HitResult hitResult;
        // Smaller tMin will has a impact on performance
        if (hit(currentRay, Math::epsilon, Math::infinity, hitResult, spheres)) {
            Float3 attenuation;
            Ray scattered;
            if (hitResult.material->scatter(currentRay, hitResult, attenuation, scattered, randState)) {
                currentAttenuation *= attenuation;
                currentRay = scattered;
            }
            else {
                return make_float3(0.0f, 0.0f, 0.0f);
            }
        }
        else {
            auto unitDirection = normalize(currentRay.direction);
            auto t = 0.5f * (unitDirection.y + 1.0f);
            auto background = lerp(make_float3(1.0f, 1.0f, 1.0f), make_float3(0.5f, 0.7f, 1.0f), t);
            return currentAttenuation * background;
        }
    }

    // exceeded recursion
    return make_float3(0.0f, 0.0f, 0.0f);
}

//CUDA_DEVICE Float3 rayColor(const Ray& ray, curandState* randState, Sphere* spheres, int32_t depth) {
//    if (depth == 0) {
//        // exceeded recursion
//        return make_float3(0.0f, 0.0f, 0.0f);
//    }
//    HitResult hitResult;
//    // Smaller tMin will has a impact on performance
//    if (hit(ray, Math::epsilon, Math::infinity, hitResult, spheres)) {
//        Float3 attenuation;
//        Ray rayScattered;
//        if (hitResult.material->scatter(ray, hitResult, attenuation, rayScattered, randState)) {
//            return attenuation * rayColor(rayScattered, randState, spheres, depth - 1);
//        }
//        else {
//            return make_float3(0.0f, 0.0f, 0.0f);
//        }
//    }
//
//    auto unitDirection = normalize(ray.direction);
//    auto t = 0.5f * (unitDirection.y + 1.0f);
//    auto background = lerp(make_float3(1.0f, 1.0f, 1.0f), make_float3(0.5f, 0.7f, 1.0f), t);
//    return background;
//}

CUDA_GLOBAL void renderInit(int32_t width, int32_t height, curandState* randState) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;
    auto index = y * width + x;

    if (index < (width * height)) {
        //Each thread gets same seed, a different sequence number, no offset
        curand_init(1984, index, 0, &randState[index]);
    }
}

//CUDA_GLOBAL void render(Canvas canvas, Camera camera, curandState* randStates, Sphere* spheres) {
//    auto x = threadIdx.x + blockDim.x * blockIdx.x;
//    auto y = threadIdx.y + blockDim.y * blockIdx.y;
//    auto width = canvas.getWidth();
//    auto height = canvas.getHeight();
//    constexpr auto samplesPerPixel = 1;
//    constexpr auto maxDepth = 5;
//    auto index = y * width + x;
//
//    if (index < (width * height)) {
//        auto color = make_float3(0.0f, 0.0f, 0.0f);
//        auto localRandState = randStates[index];
//        for (auto i = 0; i < samplesPerPixel; i++) {
//
//            auto rx = curand_uniform(&localRandState);
//            auto ry = curand_uniform(&localRandState);
//
//            auto dx = Float(x + rx) / (width - 1);
//            auto dy = Float(y + ry) / (height - 1);
//
//            auto ray = camera.getRay(dx, dy);
//            color += rayColor(ray, &localRandState, spheres);
//        }
//        // Very important!!!
//        randStates[index] = localRandState;
//        canvas.writePixel(index, color / samplesPerPixel);
//    }
//}

CUDA_GLOBAL void render(Canvas* canvas, Camera* camera, curandState* randStates, Sphere* spheres) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;
    auto width = canvas->getWidth();
    auto height = canvas->getHeight();
    constexpr auto samplesPerPixel = 1;
    constexpr auto maxDepth = 5;
    auto index = y * width + x;

    if (index < (width * height)) {
        auto color = make_float3(0.0f, 0.0f, 0.0f);
        auto localRandState = randStates[index];
        for (auto i = 0; i < samplesPerPixel; i++) {

            auto rx = curand_uniform(&localRandState);
            auto ry = curand_uniform(&localRandState);

            auto dx = Float(x + rx) / (width - 1);
            auto dy = Float(y + ry) / (height - 1);

            auto ray = camera->getRay(dx, dy);
            color += rayColor(ray, &localRandState, spheres);
        }
        // Very important!!!
        randStates[index] = localRandState;
        //canvas->writePixel(index, color / samplesPerPixel);
        canvas->accumulatePixel(index, color);
    }
}

CUDA_GLOBAL void createLambertianMaterial(Material** material, Float3 albedo, Float absorb = 1.0f) {
    (*material) = new Lambertian(albedo, absorb);
}

CUDA_GLOBAL void createMetalMaterial(Material** material, Float3 albedo, Float fuzz = 1.0f) {
    (*material) = new Metal(albedo, fuzz);
}

CUDA_GLOBAL void createDieletricMaterial(Material** material, Float indexOfRefraction = 1.5f) {
    (*material) = new Dieletric(indexOfRefraction);
}

CUDA_GLOBAL void clearBackBuffers(Canvas* canvas) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;
    auto width = canvas->getWidth();
    auto height = canvas->getHeight();

    auto index = y * width + x;

    if (index < (width * height)) {
        canvas->clearPixel(index);
    }
}

#define RESOLUTION 0

#if RESOLUTION == 0
int32_t width = 512;
int32_t height = 288;
#elif RESOLUTION == 1
int32_t width = 1024;
int32_t height = 576;
#elif RESOLUTION == 2
int32_t width = 1280;
int32_t height = 720;
#elif RESOLUTION == 3
int32_t width = 1920;
int32_t height = 1080;
#elif RESOLUTION == 4
int32_t width = 64;
int32_t height = 36;
#endif

int32_t sampleCount = 0;

Canvas* canvas = nullptr;
Camera* camera = nullptr;
Sphere* spheres = nullptr;
Material** materials[SPHERES];
curandState* randStates = nullptr;
std::shared_ptr<ImageData> imageData = nullptr;

dim3 blockSize(32, 32);
dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
              (height + blockSize.y - 1) / blockSize.y);

void initialize(int32_t width, int32_t height) {
    //Canvas canvas(width, height);
    Utils::reportGPUUsageInfo();
    canvas = createObjectPtr<Canvas>();
    canvas->initialize(width, height);
    Utils::reportGPUUsageInfo();
    //Camera camera(make_float3(-2.0f, 2.0f, 1.0f), make_float3(0.0f, 0.0f, -1.0f), make_float3(0.0f, 1.0f, 0.0f), Float(width) / height, 20.0f);
    camera = createObjectPtr<Camera>();
    camera->initialize(make_float3(-2.0f, 2.0f, 1.0f), make_float3(0.0f, 0.0f, -1.0f), make_float3(0.0f, 1.0f, 0.0f), Float(width) / height, 20.0f);

    spheres = createObjectArray<Sphere>(SPHERES);

    for (auto& material : materials) {
        material = createObjectPtr<Material*>();
    }

    createDieletricMaterial<<<1, 1>>>(materials[0], 1.5f);
    createDieletricMaterial<<<1, 1>>>(materials[1], 1.5f);
    createLambertianMaterial<<<1, 1>>>(materials[2], make_float3(0.1f, 0.2f, 0.5f));
    //createDieletricMaterial<<<1, 1>>>(materials[3], 1.5f);
    createMetalMaterial<<<1, 1>>>(materials[3], make_float3(0.8f, 0.6f, 0.2f), 0.0f);
    createLambertianMaterial<<<1, 1>>>(materials[4], make_float3(0.8f, 0.8f, 0.0f));
    gpuErrorCheck(cudaDeviceSynchronize());

    spheres[0] = { { -1.0f, 0.0f, -1.0f},   0.5f, *(materials[0]), true };
    spheres[1] = { { -1.0f, 0.0f, -1.0f }, -0.4f, *(materials[1]), false };
    spheres[2] = { {  0.0f, 0.0f, -1.0f },  0.5f, *(materials[2]), true };
    spheres[3] = { {  1.0f, 0.0f, -1.0f },  0.5f, *(materials[3]), true };
    spheres[4] = { {  0.0f, -100.5f, -1.0f }, 100.0f, *(materials[4]), true };

    auto pixelCount = width * height;
    randStates = createObjectArray<curandState>(pixelCount);

    renderInit<<<gridSize, blockSize>>>(width, height, randStates);
    gpuErrorCheck(cudaDeviceSynchronize());

    imageData = std::make_shared<ImageData>();

    imageData->width = width;
    imageData->height = height;
    imageData->channels = 3;
    imageData->size = pixelCount * 3;
}   

void clearBackBuffers() {
    clearBackBuffers<<<gridSize, blockSize>>>(canvas);
    gpuErrorCheck(cudaDeviceSynchronize());
    canvas->resetSampleCount();
}

void pathTracing() {
    if (camera->isDirty()) {
        clearBackBuffers();
        camera->resetDiryFlag();
    }

    canvas->incrementSampleCount();
    render<<<gridSize, blockSize>>>(canvas, camera, randStates, spheres);
    gpuErrorCheck(cudaDeviceSynchronize());

    imageData->data = canvas->getPixelBuffer();
}

void cleanup() {
    deleteObject(randStates);

    for (auto i = 0; i < SPHERES; i++) {
        deleteDeviceObject<<<1, 1>>>(materials[i]);
        gpuErrorCheck(cudaDeviceSynchronize());
        gpuErrorCheck(cudaFree(materials[i]));
}

    deleteObject(spheres);

    deleteObject(camera);
    canvas->uninitialize();
    deleteObject(canvas);
}

#ifndef GPU_REALTIME
int main() {
    //gpuErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    initialize(width, height);

    //gpuErrorCheck(cudaMemcpyToSymbol(constantSpheres, spheres, sizeof(Sphere) * SPHERES));
    
    GPUTimer timer("Rendering start...");
    pathTracing();
    timer.stop("Rendering elapsed time");

    canvas->writeToPNG("render.png");
    Utils::openImage(L"render.png");

    cleanup();

    return 0;
}
#endif // !GPU_REALTIME