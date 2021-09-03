
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.h"
#include "Canvas.h"
#include "GPUTimer.h"
#include "Camera.h"
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

CUDA_GLOBAL void render(Canvas canvas, Camera camera, curandState* randStates, Sphere* spheres) {
    auto x = threadIdx.x + blockDim.x * blockIdx.x;
    auto y = threadIdx.y + blockDim.y * blockIdx.y;
    auto width = camera.getImageWidth();
    auto height = camera.getImageHeight();
    constexpr auto samplesPerPixel = 100;
    constexpr auto maxDepth = 5;
    auto index = y * width + x;

    if (index < (width * height)) {
        auto color = make_float3(0.0f, 0.0f, 0.0f);
        auto localRandState = randStates[index];
        for (auto i = 0; i < samplesPerPixel; i++) {

            auto rx = 0.0f; // curand_uniform(&localRandState);
            auto ry = 0.0f; // curand_uniform(&localRandState);

            auto dx = Float(x + rx) / (width - 1);
            auto dy = Float(y + ry) / (height - 1);

            auto ray = camera.getRay(dx, dy);
            color += rayColor(ray, &localRandState, spheres);
        }
        // Very important!!!
        randStates[index] = localRandState;
        canvas.writePixel(index, color / samplesPerPixel);
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

std::string toPPM(int32_t width, int32_t height) {
    auto ppm = std::string();
    ppm.append("P3\n");
    ppm.append(std::to_string(width) + " " + std::to_string(height) + "\n");
    ppm.append(std::to_string(255) + "\n");
    return ppm;
}

void writeToPPM(const std::string& path, uint8_t* pixelBuffer, int32_t width, int32_t height) {
    auto ppm = std::ofstream(path);

    if (!ppm.is_open()) {
        std::cout << "Open file image.ppm failed.\n";
    }

    std::stringstream ss;
    ss << toPPM(width, height);

    for (auto y = height - 1; y >= 0; y--) {
        for (auto x = 0; x < width; x++) {
            auto index = y * width + x;
            auto r = uint32_t(pixelBuffer[index * 3]);
            auto g = uint32_t(pixelBuffer[index * 3 + 1]);
            auto b = uint32_t(pixelBuffer[index * 3 + 2]);
            ss << r << ' ' << g << ' ' << b << '\n';
        }
    }

    ppm.write(ss.str().c_str(), ss.str().size());

    ppm.close();
}

template<typename T>
CUDA_GLOBAL void deleteDeviceObject(T** object) {
    delete (*object);
}

#define RESOLUTION 1

int main() {
    gpuErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

#if RESOLUTION == 0
    constexpr auto width = 512;
    constexpr auto height = 384;
#elif RESOLUTION == 1
    constexpr auto width = 1280;
    constexpr auto height = 720;
#elif RESOLUTION == 2
    constexpr auto width = 1920;
    constexpr auto height = 1080;
#endif
    constexpr auto pixelCount = width * height;

    Canvas canvas(width, height);
    //auto* canvas = createObject<Canvas>();
    //canvas->initialize(width, height);

    Camera camera(width, height);
    //auto* camera = createObject<Camera>();
    //camera->initialize(width, height);

    Material** materials[SPHERES];

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

    auto* spheres = createObjectArray<Sphere>(SPHERES);

    spheres[0].center = {-1.0f, 0.0f, -1.0f};
    spheres[0].material = *(materials[0]);
    spheres[0].radius = 0.5f;
    spheres[0].bShading = true;

    spheres[1].center = { -1.0f, 0.0f, -1.0f };
    spheres[1].material = *(materials[1]);
    spheres[1].radius = -0.4f;
    spheres[1].bShading = false;

    spheres[2].center = { 0.0f, 0.0f, -1.0f };
    spheres[2].material = *(materials[2]);
    spheres[2].radius = 0.5f;
    spheres[2].bShading = true;

    spheres[3].center = { 1.0f, 0.0f, -1.0f };
    spheres[3].material = *(materials[3]);
    spheres[3].radius = 0.5f;
    spheres[3].bShading = true;

    spheres[4].center = { 0.0f, -100.5f, -1.0f };
    spheres[4].material = *(materials[4]);
    spheres[4].radius = 100.0f;
    spheres[4].bShading = true;

    gpuErrorCheck(cudaMemcpyToSymbol(constantSpheres, spheres, sizeof(Sphere) * SPHERES));

    auto* randStates = createObjectArray<curandState>(pixelCount);
    
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    renderInit<<<gridSize, blockSize>>>(width, height, randStates);
    gpuErrorCheck(cudaDeviceSynchronize());

    GPUTimer timer("Rendering start...");

    render<<<gridSize, blockSize>>>(canvas, camera, randStates, spheres);
    gpuErrorCheck(cudaDeviceSynchronize());

    timer.stop("Rendering elapsed time");

    canvas.writeToPNG("render.png");
    Utils::openImage(L"render.png");
    writeToPPM("render.ppm", canvas.getPixelBuffer(), width, height);
    deleteObject(randStates);

    for (auto i = 0; i < SPHERES; i++) {
        deleteDeviceObject<<<1, 1>>>(materials[i]);
        gpuErrorCheck(cudaDeviceSynchronize());
        gpuErrorCheck(cudaFree(materials[i]));
    }

    deleteObject(spheres);

    //deleteObject(camera);

    canvas.uninitialize();
    //deleteObject(canvas);

    return 0;
}
