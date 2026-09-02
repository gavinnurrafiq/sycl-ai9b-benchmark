# SYCL AI 9B Benchmark

`sycl-ai-benchmark` is a lightweight C++ benchmarking utility built with SYCL 2020. It evaluates whether a target GPU hardware has sufficient VRAM capacity and compute throughput to handle Large Language Models (LLMs) with 9 billion (9B) parameters—such as Llama 3 8B/9B, Gemma 9B, or Mistral 7B/9B class models.

## 📖 Overview

Running 9B parameter LLMs requires substantial VRAM capacity and high-precision floating-point performance. This benchmark performs two key evaluations:

1. **VRAM Capacity Check & Stress Test:** Dynamically queries hardware limits and allocates large-scale FP16 (`sycl::half`) Unified Shared Memory (USM) buffers directly on the GPU to test memory limits (~1.6 GB active buffer space).
2. **FP16 Compute Throughput Test:** Executes 50 iterations of parallel fused multiply-add (FMA) kernel dispatches across hundreds of millions of elements to measure peak FP16 TFLOPS performance.

---

## 🛠️ Prerequisites

- **Compiler:** Intel® DPC++/C++ Compiler (`icpx`).
- **Toolkit:** Intel® oneAPI Base Toolkit (with SYCL support).
- **Hardware:** SYCL-compatible GPU runtime.

---

## 🚀 Building & Running

### Step 1: Initialize Intel oneAPI Environment
Open your terminal or command prompt and source the oneAPI environment variables:

```cmd
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

### Step 2: Compile the Project
Compile the source code using `icpx` with SYCL enabled (`-fsycl`):

```cmd
icpx -fsycl main.cpp -o sycl_ai_benchmark.exe
```

### Step 3: Execute the Benchmark
Run the generated executable:

```cmd
.\sycl_ai_benchmark.exe
```

---

## 💡 How It Works

1. **Device Selection (`sycl::gpu_selector_v`):** Automatically binds to an available GPU device and queries total global memory (VRAM).
2. **Device Memory Allocation (`sycl::malloc_device`):** Allocates three contiguous FP16 device buffers (`d_A`, `d_B`, `d_C`) totalling over 268 million elements (`MATRIX_M * MATRIX_N`).
3. **Parallel Kernel Dispatch (`h.parallel_for`):** Offloads element-wise fused operations (`d_C[idx] = d_A[idx] * d_B[idx] + d_C[idx]`) to execution threads on the GPU.
4. **Performance Calculation:** Calculates execution time using standard C++ high-resolution timers and computes effective FP16 throughput in TFLOPS ($2.0 \times \text{elements} \times \text{iterations} / \text{time}$).
