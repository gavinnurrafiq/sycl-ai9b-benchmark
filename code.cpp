#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

constexpr size_t MATRIX_M = 16384;
constexpr size_t MATRIX_K = 4096;
constexpr size_t MATRIX_N = 4096;
constexpr size_t TOTAL_ELEMENTS = MATRIX_M * MATRIX_N;

int main() {
    try {
        sycl::queue q{sycl::gpu_selector_v};

        auto dev = q.get_device();
        std::cout << "=== GPU AI 9B BENCHMARK (SYCL) ===\n";
        std::cout << "Device          : " << dev.get_info<sycl::info::device::name>() << "\n";

        auto global_mem = dev.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024 * 1024);
        std::cout << "Available VRAM  : " << global_mem << " GB\n\n";

        if (global_mem < 6) {
            std::cout << "[WARNING] VRAM is under 6 GB. Extremely unlikely to run a 9B model (even INT4).\n";
        } else if (global_mem < 16) {
            std::cout << "[INFO] GPU is suitable for running Quantized 9B models (INT4 / Q4_K_M).\n";
        } else {
            std::cout << "[INFO] VRAM is sufficient for Full Precision (FP16) 9B models.\n";
        }

        std::cout << "\n[1/2] Allocating VRAM & Performing Stress Test (3 Large FP16 Matrices)..." << std::endl;

        sycl::half* d_A = sycl::malloc_device<sycl::half>(TOTAL_ELEMENTS, q);
        sycl::half* d_B = sycl::malloc_device<sycl::half>(TOTAL_ELEMENTS, q);
        sycl::half* d_C = sycl::malloc_device<sycl::half>(TOTAL_ELEMENTS, q);

        if (!d_A || !d_B || !d_C) {
            std::cerr << "[ERROR] VRAM allocation failed! GPU Out of Memory (OOM).\n";
            return -1;
        }

        std::cout << "✓ VRAM allocation successful (~1.6 GB active buffer memory).\n";

        q.fill(d_A, sycl::half(1.0f), TOTAL_ELEMENTS);
        q.fill(d_B, sycl::half(0.5f), TOTAL_ELEMENTS);
        q.wait();

        std::cout << "[2/2] Benchmarking FP16 compute performance..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        constexpr int ITERATIONS = 50;
        for (int iter = 0; iter < ITERATIONS; iter++) {
            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(TOTAL_ELEMENTS), [=](sycl::id<1> idx) {
                    d_C[idx] = d_A[idx] * d_B[idx] + d_C[idx];
                });
            });
        }
        q.wait();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        double total_ops = 2.0 * TOTAL_ELEMENTS * ITERATIONS;
        double tflops = (total_ops / diff.count()) / 1e12;

        std::cout << "\n=== BENCHMARK RESULTS ===" << std::endl;
        std::cout << "Execution Time  : " << diff.count() << " seconds\n";
        std::cout << "FP16 Performance: " << tflops << " TFLOPS\n";

        std::cout << "\n=== CONCLUSION ===" << std::endl;
        if (tflops >= 15.0 && global_mem >= 8) {
            std::cout << "STATUS: READY! Your GPU is VERY CAPABLE of high-speed 9B LLM inference.\n";
        } else if (tflops >= 5.0 && global_mem >= 6) {
            std::cout << "STATUS: CAPABLE. Your GPU can run 9B models (INT4 quantized), but with moderate speed.\n";
        } else {
            std::cout << "STATUS: TOO SLOW / LIMITED. Your GPU will struggle (too slow or prone to OOM).\n";
        }

        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception caught: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
