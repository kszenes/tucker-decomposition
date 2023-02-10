#ifndef TIMER_H
#define TIMER_H

#include <cuda_runtime.h>
#include <chrono>

struct CPUTimer
{
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    float seconds() {
        stop_ = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> diff = stop_ - start_;
        return diff.count();
    }
    private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_, stop_;
};

struct GPUTimer
{
    GPUTimer() 
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~GPUTimer() 
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() 
    {
        cudaEventRecord(start_, 0);
    }

    float seconds() 
    {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time * 1e-3;
    }
    private:
    cudaEvent_t start_, stop_;
};


#endif /* TIMER_H */
