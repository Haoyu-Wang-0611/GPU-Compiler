// stencil1d_general.cu — 通用半径 1D stencil 穷举载具（holdout S1/S2/S3/S7）
//
// 语义：out[i] = (1/(2R+1)) * sum_{d=-R..R} in[i+d]，i ∈ [R, N-R)
// 载具形态与 heat1d_oracle 同源：cp.async 多缓冲 + grid-stride tile；
//   smem 口径按模型 legacy_base：每份 (T+2R) 读 + T 写 → **(S+1)(2T+2R) floats**，
//   其中只实际使用 A 的 (S+1)(T+2R) 段，其余为容量占位（保证 oracle 可行集 = 模型可行集）。
//
// 用法（单配置）：
//   ./stencil1d_general N T S R
//   例：./stencil1d_general 350000 512 1 2
//
// 输出行（供 tools/four_arms.py 解析）：
//   T=512      S=1  smem=  8.0KB  实测:       5.0 us  PASS
//
// 编译：
//   nvcc -O3 -arch=sm_89 stencil1d_general.cu -o stencil1d_general
//
// 注意：N 不要求 4 的倍数（4B cp.async）；R=1 时与 heat1d 同量级。

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define NUM_SMS 128
#define SMEM_CAP 101376
#define BLOCK 256

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// 预取 tile k 的 A 窗口 [k*T, k*T+SEG) 到 dst（device 函数，避免 lambda 的编译开关问题）
__device__ __forceinline__ void prefetch_tile(const float* __restrict__ in,
                                              float* dst, int start, int n, int SEG) {
    for (int i = threadIdx.x; i < SEG; i += blockDim.x) {
        int g = start + i;
        if (g < n) __pipeline_memcpy_async(&dst[i], &in[g], 4);
        else       dst[i] = 0.f;
    }
}

// ---- 通用半径 1D stencil：out[R + o] = mean(in[o .. o+2R]) ----
__global__ void stencil1d_async(const float* __restrict__ in,
                                float* __restrict__ out,
                                int n, int T, int S, int R) {
    extern __shared__ float smem[];          // (S+1)*(2T+2R) floats（只用前 (S+1)(T+2R)）
    const int SEG = T + 2 * R;               // 每 tile 读窗口
    const int M = n - 2 * R;                 // 输出个数（相对 o ∈ [0,M)）
    const int NTILE = (M + T - 1) / T;
    const int G = gridDim.x;
    const float coef = 1.0f / (float)(2 * R + 1);

    // prologue：本 block 的前 S 个 tile
    for (int j = 0; j < S; ++j) {
        int k = blockIdx.x + j * G;
        if (k >= NTILE) break;
        prefetch_tile(in, smem + j * SEG, k * T, n, SEG);
        __pipeline_commit();
    }

    int my_tiles = (NTILE - blockIdx.x + G - 1) / G;
    for (int c = 0; c < my_tiles; ++c) {
        int k = blockIdx.x + c * G;
        int slot = c % (S + 1);
        __pipeline_wait_prior(S - 1);
        __syncthreads();

        float* s = smem + slot * SEG;
        int base = k * T;
        for (int t = threadIdx.x; t < T; t += blockDim.x) {
            int o = base + t;                 // 相对输出下标
            if (o >= M) continue;
            float sum = 0.f;
            for (int d = 0; d < SEG - T + 1; ++d)   // d = 0..2R
                sum += s[t + d];
            out[R + o] = sum * coef;
        }
        __syncthreads();                       // 读完才允许覆写本槽

        int kf = k + S * G;
        if (kf < NTILE) {
            int fslot = (c + S) % (S + 1);
            prefetch_tile(in, smem + fslot * SEG, kf * T, n, SEG);
        }
        __pipeline_commit();
    }
}

// ---- 正确性：out[R+o] == mean(in[o..o+2R]) ----
__global__ void check_kernel(const float* __restrict__ in,
                             const float* __restrict__ out,
                             int n, int R, int* err) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int M = n - 2 * R;
    if (o >= M) return;
    float sum = 0.f;
    for (int d = 0; d < 2 * R + 1; ++d) sum += in[o + d];
    float ref = sum / (float)(2 * R + 1);
    float got = out[R + o];
    if (fabsf(got - ref) > 1e-4f * (fabsf(ref) + 1e-6f)) atomicAdd(err, 1);
}

int main(int argc, char** argv) {
    int N = 350000, T = 512, S = 1, R = 1;
    if (argc >= 5) { N = atoi(argv[1]); T = atoi(argv[2]); S = atoi(argv[3]); R = atoi(argv[4]); }
    else if (argc != 1) { printf("[参数] 用法: ./stencil1d_general N T S R\n"); return 1; }
    if (N <= 2 * R) { printf("[参数] N 必须 > 2R\n"); return 1; }
    if (T <= 0 || S <= 0 || R <= 0) { printf("[参数] T/S/R 必须 > 0\n"); return 1; }

    long long smem = (long long)(S + 1) * (2LL * T + 2LL * R) * 4;   // 模型口径
    long long seg  = (long long)(S + 1) * (T + 2LL * R) * 4;         // 实际使用的 A 段
    printf("[参数] N=%d R=%d T=%d S=%d smem_model=%.1fKB seg=%.1fKB\n",
           N, R, T, S, smem / 1024.0, seg / 1024.0);
    if (smem > SMEM_CAP) { printf("T=%-8d S=%d  smem=%6.1fKB 超限跳过\n", T, S, smem / 1024.0); return 0; }

    size_t bytes = (size_t)N * sizeof(float);
    std::vector<float> hA(N), hRef(N, 0.f);
    for (int i = 0; i < N; ++i) hA[i] = (float)((i * 17) % 101) * 0.01f;
    for (int o = 0; o < N - 2 * R; ++o) {
        double sum = 0;
        for (int d = 0; d < 2 * R + 1; ++d) sum += hA[o + d];
        hRef[R + o] = (float)(sum / (2 * R + 1));
    }

    float *dA, *dOut;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dOut, bytes));
    CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dOut, 0, bytes));
    CHECK(cudaFuncSetAttribute(stencil1d_async,
          cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_CAP));

    int M = N - 2 * R;
    int NTILE = (M + T - 1) / T;
    int grid = std::min(NUM_SMS, NTILE);
    size_t dyn = (size_t)(S + 1) * (2LL * T + 2LL * R) * sizeof(float);  // 模型口径（含未用的 B 占位）

    // 正确性（每个配置一次）
    stencil1d_async<<<grid, BLOCK, dyn>>>(dA, dOut, N, T, S, R);
    { cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) { printf("T=%-8d S=%d  launch FAIL: %s —— 跳过\n",
                                     T, S, cudaGetErrorString(e)); return 0; } }
    CHECK(cudaDeviceSynchronize());
    int* dErr; CHECK(cudaMalloc(&dErr, 4)); CHECK(cudaMemset(dErr, 0, 4));
    check_kernel<<<(M + 255) / 256, 256>>>(dA, dOut, N, R, dErr);
    int hErr = -1; CHECK(cudaMemcpy(&hErr, dErr, 4, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dErr));

    // 计时：3 rep 取最优
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));
    double best = 1e18;
    for (int rep = 0; rep < 3; ++rep) {
        float ms = 0;
        CHECK(cudaEventRecord(ev0));
        stencil1d_async<<<grid, BLOCK, dyn>>>(dA, dOut, N, T, S, R);
        CHECK(cudaEventRecord(ev1));
        CHECK(cudaEventSynchronize(ev1));
        CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        if (ms < best) best = ms;
    }
    { cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) { printf("T=%-8d S=%d  计时 launch FAIL: %s —— 跳过\n",
                                     T, S, cudaGetErrorString(e)); return 0; } }
    double us = best * 1000.0;
    printf("T=%-8d S=%d  smem=%5.1fKB  实测: %12.1f us  %s\n",
           T, S, smem / 1024.0, us, hErr == 0 ? "PASS" : "FAIL");
    CHECK(cudaFree(dA)); CHECK(cudaFree(dOut));
    return 0;
}
