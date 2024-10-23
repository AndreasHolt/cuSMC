#include <iostream>
#include <cuda_runtime.h>  // Include CUDA runtime header

__global__ void add(int n, float* a, float* b, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread index
    if (i < n) {
        c[i] = a[i] + b[i]; // Perform addition
    }
}

int lol() {
    int n = 10;

    // Allocate memory on the host
    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_c = new float[n];

    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
        h_c[i] = 0;
    }

    // Allocate memory on the device
    float *d_a, *d_b, *d_c;
    // &d_a is the address of the pointer variable. So we can pass the pointer by reference to the function. This allows cudaMalloc to change where d_a points to
    // essentially &d_a is a pointer to a pointer of a float
    cudaMalloc(&d_a, n * sizeof(float)); // Allocate device memory for a
    cudaMalloc(&d_b, n * sizeof(float)); // Allocate device memory for b
    cudaMalloc(&d_c, n * sizeof(float)); // Allocate device memory for c

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    add<<<(n + 255) / 256, 256>>>(n, d_a, d_b, d_c); // Calculate grid and block sizes
    cudaDeviceSynchronize(); // Wait for the kernel to finish

    // Copy result back from device to host
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < n; i++) {
        std::cout << h_c[i] << " "; // Print results
    }
    std::cout << "Lol, World!" << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}

