#include <iostream>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "lif_kernel.h"

__device__ float sigmoidSurrogate(
    const float scalarInput,
    const float threshold
){
    const float alpha = 4.0f;
    return alpha / 2.0f / (1.0f + coshf(alpha * (scalarInput - threshold)));
}

__device__ float hardSigmoidSurrogate(
    const float scalarInput,
    const float threshold,
    const float lens
){
    return fabsf(scalarInput - threshold) < lens ? 1.0f : 0.0f;
}

__global__ void fusedForwardLIFKernel_V(
    const float* tX,
    float* V,
    float* tY,
    const int timeStep,
    const size_t tensorSize,
    const float decay,
    const float threshold,
    const float rest
){
    const size_t tensorIndex = (size_t)blockIdx.x * blockDim.x + (size_t)threadIdx.x;
    if (tensorIndex >= tensorSize) return;
    float v = rest;
    float y = (v >= threshold) ? 1.0f : 0.0f;
    for (int t = 0; t < timeStep; t++){
        const size_t pos = t * tensorSize + tensorIndex;
        v = decay * v * (1.0f - y) + rest * y + tX[pos];
        y = (v >= threshold) ? 1.0f : 0.0f;
        tY[pos] = y;
    }
    V[tensorIndex] = v;
}

__global__ void fusedForwardLIFKernel_tV(
    const float* tX,
    float* tV,
    float* tY,
    const int timeStep,
    const size_t tensorSize,
    const float decay,
    const float threshold,
    const float rest
){
    const size_t tensorIndex = (size_t)blockIdx.x * blockDim.x + (size_t)threadIdx.x;
    if (tensorIndex >= tensorSize) return;
    float v = rest;
    float y = (v >= threshold) ? 1.0f : 0.0f;
    for (int t = 0; t < timeStep; t++){
        const size_t pos = t * tensorSize + tensorIndex;
        v = decay * v * (1.0f - y) + rest * y + tX[pos];
        y = (v >= threshold) ? 1.0f : 0.0f;
        tV[pos] = v;
        tY[pos] = y;
    }
}

__global__ void fusedBackwardLIFKernel(
    const float* gtY,
    float* gtX,
    const float* tY,
    const float* tV,
    const int timeStep,
    const size_t tensorSize,
    const float decay,
    const float threshold
){
    const size_t tensorIndex = (size_t)blockIdx.x * blockDim.x + (size_t)threadIdx.x;
    if (tensorIndex >= tensorSize) return;
    float l2v = 0.0f;
    const float lens = 0.5f;
    for (int t = timeStep - 1; t >= 0; t--){
        const size_t pos = t * tensorSize + tensorIndex;
        const float y2v = hardSigmoidSurrogate(tV[pos], threshold, lens);
        const float l2y = gtY[pos];
        const float v2v = decay * (1.0f - tY[pos] - tV[pos] * y2v);
        l2v = l2y * y2v + l2v * v2v;
        gtX[pos] = l2v;
    }
}

void launch_fusedForwardLIFKernel(
    const float* tX,
    float* V,
    float* tY,
    const int timeStep,
    const size_t tensorSize,
    const float decay,
    const float threshold,
    const float rest,
    bool use_tV
){
    cudaError_t err;
    int gridSize{}, blockSize{};
    if (use_tV){
        err = cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, fusedForwardLIFKernel_tV);
    } else {
        err = cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, fusedForwardLIFKernel_V);
    }
    if (err != cudaSuccess){
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
    if ((size_t)gridSize * blockSize < tensorSize){
        gridSize = (int)((tensorSize - 1) / blockSize + 1);
    }
    if (use_tV){
        fusedForwardLIFKernel_tV<<<gridSize, blockSize>>>(tX, V, tY, timeStep, tensorSize, decay, threshold, rest);
    } else {
        fusedForwardLIFKernel_V<<<gridSize, blockSize>>>(tX, V, tY, timeStep, tensorSize, decay, threshold, rest);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess){
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
}

void launch_fusedBackwardLIFKernel(
    const float* gtY,
    float* gtX,
    const float* tY,
    const float* tV,
    const int timeStep,
    const size_t tensorSize,
    const float decay,
    const float threshold
){
    cudaError_t err;
    int gridSize{}, blockSize{};
    err = cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, fusedBackwardLIFKernel);
    if (err != cudaSuccess){
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
    if ((size_t)gridSize * blockSize < tensorSize){
        gridSize = (int)((tensorSize - 1) / blockSize + 1);
    }
    fusedBackwardLIFKernel<<<gridSize, blockSize>>>(gtY, gtX, tY, tV, timeStep, tensorSize, decay, threshold);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
}
