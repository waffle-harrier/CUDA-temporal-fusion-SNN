#include <iostream>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ternary_lif_kernel.h"

__device__ float hardTernarySurrogate(
    const float scalarInput,
    const float threshold,
    const float lens
){
    float diff_pos = fabsf(scalarInput - threshold);
    float diff_neg = fabsf(scalarInput + threshold);
    return (diff_pos < lens || diff_neg < lens) ? 1.0f : 0.0f;
}

__global__ void fusedForwardTernaryLIFKernel_V(
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
    float y = 0.0f;
    for (int t = 0; t < timeStep; t++){
        const size_t pos = t * tensorSize + tensorIndex;
        v = decay * v * (1.0f - fabsf(y)) + tX[pos];
        if (v >= threshold){
            y = 1.0f;
        } else if (v <= -threshold){
            y = -1.0f;
        } else {
            y = 0.0f;
        }
        tY[pos] = y;
    }
    V[tensorIndex] = v;
}

__global__ void fusedForwardTernaryLIFKernel_tV(
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
    float y = 0.0f;
    for (int t = 0; t < timeStep; t++){
        const size_t pos = t * tensorSize + tensorIndex;
        v = decay * v * (1.0f - fabsf(y)) + tX[pos];
        if (v >= threshold){
            y = 1.0f;
        } else if (v <= -threshold){
            y = -1.0f;
        } else {
            y = 0.0f;
        }
        tV[pos] = v;
        tY[pos] = y;
    }
}

__global__ void fusedBackwardTernaryLIFKernel(
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
    const float lens = 0.25f;
    for (int t = timeStep - 1; t >= 0; t--){
        const size_t pos = t * tensorSize + tensorIndex;
        const float y2v = hardTernarySurrogate(tV[pos], threshold, lens);
        const float l2y = gtY[pos];
        const float v2v = decay * (1.0f - fabsf(tY[pos]));
        l2v = l2y * y2v + l2v * v2v;
        gtX[pos] = l2v;
    }
}

void launch_fusedForwardTernaryLIFKernel(
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
        err = cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, fusedForwardTernaryLIFKernel_tV);
    } else {
        err = cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, fusedForwardTernaryLIFKernel_V);
    }
    if (err != cudaSuccess){
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
    if ((size_t)gridSize * blockSize < tensorSize){
        gridSize = (int)((tensorSize - 1) / blockSize + 1);
    }
    if (use_tV){
        fusedForwardTernaryLIFKernel_tV<<<gridSize, blockSize>>>(tX, V, tY, timeStep, tensorSize, decay, threshold, rest);
    } else {
        fusedForwardTernaryLIFKernel_V<<<gridSize, blockSize>>>(tX, V, tY, timeStep, tensorSize, decay, threshold, rest);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess){
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
}

void launch_fusedBackwardTernaryLIFKernel(
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
    err = cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, fusedBackwardTernaryLIFKernel);
    if (err != cudaSuccess){
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
    if ((size_t)gridSize * blockSize < tensorSize){
        gridSize = (int)((tensorSize - 1) / blockSize + 1);
    }
    fusedBackwardTernaryLIFKernel<<<gridSize, blockSize>>>(gtY, gtX, tY, tV, timeStep, tensorSize, decay, threshold);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
    }
}
