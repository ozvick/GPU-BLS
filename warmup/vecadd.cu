#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                cudaGetErrorString(err), __FILE__, __LINE__);          \
        exit(1);                                                       \
    } } while (0)

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 24;
    const size_t bytes = n * sizeof(float);

    float *h_a = (float*)malloc(bytes), *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes), *h_ref = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { h_a[i] = 0.5f * (i % 1000); h_b[i] = 1.0f; }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) h_ref[i] = h_a[i] + h_b[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));

    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;   // round up

    vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);   // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(e1));
    vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e2));

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e3));
    CUDA_CHECK(cudaEventSynchronize(e3));

    float h2d_ms, kern_ms, d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&kern_ms, e1, e2));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, e2, e3));

    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(h_c[i] - h_ref[i]) > 1e-5f) {
            if (errors == 0) printf("first mismatch at i=%d: got %f, expected %f\n",
                                    i, h_c[i], h_ref[i]);
            errors++;
        }
    }

    printf("n = %d, blocks = %d, threads = %d\n", n, blocks, threads);
    printf("CPU loop:        %8.3f ms\n", cpu_ms);
    printf("GPU kernel:      %8.3f ms\n", kern_ms);
    printf("Copy to GPU:     %8.3f ms\n", h2d_ms);
    printf("Copy from GPU:   %8.3f ms\n", d2h_ms);
    printf("GPU incl copies: %8.3f ms\n", h2d_ms + kern_ms + d2h_ms);
    printf("%s (%d mismatches)\n", errors ? "FAIL" : "PASS", errors);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c); free(h_ref);
    return 0;
}
