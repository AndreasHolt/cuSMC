// #define GPU __device__
// #define CPU __host__
// #define GLOBAL __global__
// #define IS_GPU __CUDACC__
//
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <curand.h>
// #include <curand_kernel.h>
//
// #ifdef __CUDACC__
// #define cuda_SYNCTHREADS() __syncthreads()
// #else
// #define cuda_SYNCTHREADS()
// #endif


#ifndef MACRO_H
#define MACRO_H


#include <string>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

//HACK TO MAKE CPU WORK!
#define QUALIFIERS static __forceinline__ __host__ __device__
#include <curand_kernel.h>
#undef QUALIFIERS
//HACK SLUT

//TODO: Check this

// #include "device_atomic_functions.h"
// #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// #else
// __device__ double atomicAdd(double* a, double b) { return b; }
// #endif

#ifndef CUDA_ATOMIC_DOUBLE_H
#define CUDA_ATOMIC_DOUBLE_H

#include "device_atomic_functions.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Use the built-in atomicAdd for double precision
#else
__device__ __forceinline__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#endif // CUDA_ATOMIC_DOUBLE_H

#define GPU __device__
#define CPU __host__
#define GLOBAL __global__
#define IS_GPU __CUDACC__

#define DBL_MAX 1.7976931348623158e+308 //max 64 bit double value
// #define DBL_EPSILON 2.2204460492503131e-016 // smallest such that 1.0+DBL_EPSILON != 1.0
#define DBL_EPSILON (0.00001)

//While loop done to enfore ; after macro call. See:
//https://stackoverflow.com/a/61363791/17430854
#define CUDA_CHECK(x)             \
do{                          \
if ((x) != cudaSuccess) {    \
throw std::runtime_error(std::string("cuda error ") + std::to_string(x) + " in file '" + __FILE__ + "' on line "+  std::to_string(__LINE__)); \
}                             \
}while(0)


__host__ __device__ __forceinline__ void cuda_syncthreads_() {
#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define cuda_SYNCTHREADS()
#endif
}

#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define cuda_SYNCTHREADS()
#endif

#endif
