// ============================================================
// bufprobe.cu — 深度甜点归因探针（只读实验，不改 L1Feas / 不改 sgemm_tiled.cu）
//
// 要回答的问题
//   3mm 的 128×128/TK16 上，S=3 是甜点（244 µs），S=4 反而慢 35%（330 µs）——
//   而 ptxas 显示两者寄存器都是 167、驻留都是 1，所以"驻留台阶"解释不成立。
//   sgemm_tiled 里 S 同时控制两件事：
//       缓冲份数 B = S + 1      （smem 足迹）
//       预取距离 D = S          （wait_prior(D-1)、prefetch kc+D）
//   本探针把 B 与 D 拆成两个独立模板参数，再叠一个"缓冲间 padding"，
//   用同一份计算核跑 (B,D,pad) 网格，看拐点跟谁走。
//
// 编译 / 运行
//   nvcc -O3 -arch=sm_89 -Xptxas -v -o bufprobe bufprobe.cu 2> bufprobe_ptxas.log
//   CUDA_VISIBLE_DEVICES=1 ./bufprobe 512      # 复现原反常（16 个活跃 tile）
//   CUDA_VISIBLE_DEVICES=1 ./bufprobe 1024     # 更多块（64 个活跃 tile），看是否只是低占用伪象
//
// 判读
//   若"慢 35%"只跟 D 走、与 B 无关      → 预取深度本身的问题（wait_prior/流水账本）
//   若只跟 B 走                        → smem 足迹/对齐问题
//   若加 padding 就消失                → 缓冲间固定 16KB 步长导致的分区/set 冲突
//   若 1024 上不复现                    → 低占用伪象，只在 16 块时出现
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define NUM_SMS 128
#define SMEM_CAP (99 * 1024)

// ---------------- 计算核：B 份缓冲、预取距离 D、缓冲间 padding ----------------
template <int BM, int BN, int TK, int B, int D, int PAD>
__global__ void sgemm_bd(const float* __restrict__ A, const float* __restrict__ Bm,
                         float* __restrict__ C, int M, int N, int K) {
    static_assert(D < B, "prefetch distance must be < buffer count");
    constexpr int TM = 4, TN = 4;
    constexpr int TP = (BM / TM) * (BN / TN) / 256;
    constexpr int SA = BM * TK + PAD;          // A 缓冲步长（float）
    constexpr int SB = TK * BN + PAD;          // B 缓冲步长（float）

    extern __shared__ float smem[];
    float* smA = smem;
    float* smB = smem + (size_t)B * SA;

    const int G = gridDim.x;
    const int tilesN = (N + BN - 1) / BN;
    const int total_tiles = ((M + BM - 1) / BM) * tilesN;
    const int nchunks = (K + TK - 1) / TK;
    const int tid = threadIdx.x;

    float acc[TP][TM][TN];
#pragma unroll
    for (int p = 0; p < TP; p++)
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) acc[p][i][j] = 0.f;

    const int my_tiles = (total_tiles - blockIdx.x + G - 1) / G;

    for (int c = 0; c < my_tiles; c++) {
        const int t = blockIdx.x + c * G;
        const int bm0 = (t / tilesN) * BM;
        const int bn0 = (t % tilesN) * BN;

        // prologue：预取前 D 个 chunk（D < B，槽位不会与首个计算槽冲突）
        for (int s = 0; s < D; s++) {
            if (s < nchunks) {
                float* pA = smA + (size_t)s * SA;
                float* pB = smB + (size_t)s * SB;
                for (int i = tid; i < BM * TK; i += 256) {
                    int r = i / TK, cc = i % TK;
                    int gr = bm0 + r, gc = s * TK + cc;
                    if (gr < M && gc < K)
                        __pipeline_memcpy_async(pA + i, A + (size_t)gr * K + gc, sizeof(float));
                    else pA[i] = 0.f;
                }
                for (int i = tid; i < TK * BN; i += 256) {
                    int r = i / BN, cc = i % BN;
                    int gr = s * TK + r, gc = bn0 + cc;
                    if (gr < K && gc < N)
                        __pipeline_memcpy_async(pB + i, Bm + (size_t)gr * N + gc, sizeof(float));
                    else pB[i] = 0.f;
                }
            }
            __pipeline_commit();
        }

        for (int kc = 0; kc < nchunks; kc++) {
            const int slot = kc % B;
            __pipeline_wait_prior(D - 1);
            __syncthreads();

            const float* pA = smA + (size_t)slot * SA;
            const float* pB = smB + (size_t)slot * SB;
#pragma unroll
            for (int p = 0; p < TP; p++) {
                const int gidx = p * 256 + tid;
                const int trow = (gidx / (BN / TN)) * TM;
                const int tcol = (gidx % (BN / TN)) * TN;
                for (int k = 0; k < TK; k++) {
                    const float a0 = pA[(trow + 0) * TK + k];
                    const float a1 = pA[(trow + 1) * TK + k];
                    const float a2 = pA[(trow + 2) * TK + k];
                    const float a3 = pA[(trow + 3) * TK + k];
                    const float b0 = pB[k * BN + tcol + 0];
                    const float b1 = pB[k * BN + tcol + 1];
                    const float b2 = pB[k * BN + tcol + 2];
                    const float b3 = pB[k * BN + tcol + 3];
                    acc[p][0][0] += a0 * b0; acc[p][0][1] += a0 * b1;
                    acc[p][0][2] += a0 * b2; acc[p][0][3] += a0 * b3;
                    acc[p][1][0] += a1 * b0; acc[p][1][1] += a1 * b1;
                    acc[p][1][2] += a1 * b2; acc[p][1][3] += a1 * b3;
                    acc[p][2][0] += a2 * b0; acc[p][2][1] += a2 * b1;
                    acc[p][2][2] += a2 * b2; acc[p][2][3] += a2 * b3;
                    acc[p][3][0] += a3 * b0; acc[p][3][1] += a3 * b1;
                    acc[p][3][2] += a3 * b2; acc[p][3][3] += a3 * b3;
                }
            }
            __syncthreads();

            const int nk = kc + D;
            if (nk < nchunks) {
                const int ns = nk % B;
                float* qA = smA + (size_t)ns * SA;
                float* qB = smB + (size_t)ns * SB;
                for (int i = tid; i < BM * TK; i += 256) {
                    int r = i / TK, cc = i % TK;
                    int gr = bm0 + r, gc = nk * TK + cc;
                    if (gr < M && gc < K)
                        __pipeline_memcpy_async(qA + i, A + (size_t)gr * K + gc, sizeof(float));
                    else qA[i] = 0.f;
                }
                for (int i = tid; i < TK * BN; i += 256) {
                    int r = i / BN, cc = i % BN;
                    int gr = nk * TK + r, gc = bn0 + cc;
                    if (gr < K && gc < N)
                        __pipeline_memcpy_async(qB + i, Bm + (size_t)gr * N + gc, sizeof(float));
                    else qB[i] = 0.f;
                }
            }
            __pipeline_commit();
        }

        __pipeline_wait_prior(0);
        __syncthreads();

#pragma unroll
        for (int p = 0; p < TP; p++) {
            const int gidx = p * 256 + tid;
            const int trow = (gidx / (BN / TN)) * TM;
            const int tcol = (gidx % (BN / TN)) * TN;
#pragma unroll
            for (int i = 0; i < TM; i++) {
                const int gr = bm0 + trow + i;
                if (gr >= M) continue;
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    const int gc = bn0 + tcol + j;
                    if (gc < N) C[(size_t)gr * N + gc] = acc[p][i][j];
                }
            }
        }
#pragma unroll
        for (int p = 0; p < TP; p++)
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) acc[p][i][j] = 0.f;
    }
}

__global__ void naive(const float* __restrict__ A, const float* __restrict__ Bm,
                      float* __restrict__ C, int M, int N, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * N, stride = (long long)gridDim.x * blockDim.x;
    for (; i < total; i += stride) {
        int r = (int)(i / N), c = (int)(i % N);
        float s = 0.f;
        for (int k = 0; k < K; k++) s += A[(size_t)r * K + k] * Bm[(size_t)k * N + c];
        C[i] = s;
    }
}

__global__ void errk(const float* C, const float* R, long long n, double* s2, double* r2) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long st = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += st) {
        double d = (double)C[i] - (double)R[i];
        atomicAdd(s2, d * d);
        atomicAdd(r2, (double)R[i] * (double)R[i]);
    }
}

template <int BM, int BN, int TK, int B, int D, int PAD>
double bench(const float* dA, const float* dB, float* dC, int M, int N, int K) {
    constexpr int smem = (B * (BM * TK + PAD) + B * (TK * BN + PAD)) * 4;
    if (smem > SMEM_CAP) return -2.0;                   // 超出上限，跳过
    cudaError_t e = cudaFuncSetAttribute(sgemm_bd<BM, BN, TK, B, D, PAD>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_CAP);
    if (e != cudaSuccess) return -1.0;
    dim3 grid(NUM_SMS), block(256);
    for (int w = 0; w < 3; w++)
        sgemm_bd<BM, BN, TK, B, D, PAD><<<grid, block, smem>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) return -1.0;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int it = 0; it < 20; it++)
        sgemm_bd<BM, BN, TK, B, D, PAD><<<grid, block, smem>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    if (cudaGetLastError() != cudaSuccess) return -1.0;
    return (double(ms) / 20.0) * 1000.0;
}

struct Case { int B, D, PAD; const char* note; };
static const Case CASES[] = {
    {2, 1, 0, "= 原 S=1（2 缓冲 / 预取 1）"},
    {3, 2, 0, "= 原 S=2"},
    {4, 3, 0, "= 原 S=3（3mm 甜点）"},
    {5, 4, 0, "= 原 S=4（3mm 反常慢 35%）"},
    {3, 1, 0, "固定 D=1，B 3"},
    {4, 1, 0, "固定 D=1，B 4"},
    {4, 2, 0, "固定 D=2，B 4"},
    {5, 2, 0, "固定 D=2，B 5"},
    {6, 2, 0, "固定 D=2，B 6"},
    {5, 3, 0, "固定 D=3，B 5"},
    {6, 3, 0, "固定 D=3，B 6"},
    {6, 4, 0, "固定 D=4，B 6"},
    {4, 3, 32, "甜点 + 128B padding"},
    {4, 3, 64, "甜点 + 256B padding"},
    {5, 4, 32, "反常点 + 128B padding"},
    {5, 4, 64, "反常点 + 256B padding"},
    {1, 0, 0, "（占位，不用）"},
};

int main(int argc, char** argv) {
    int SZ = (argc > 1) ? atoi(argv[1]) : 1024;
    const int M = SZ, N = SZ, K = SZ;

    // ---- 0. 设备规格（一条都别猜）----
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    printf("[dev] smem/SM              = %d B (%.1f KB)\n", v, v / 1024.0);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxRegistersPerMultiprocessor, 0);
    printf("[dev] regs/SM              = %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxBlocksPerMultiprocessor, 0);
    printf("[dev] blocks/SM            = %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    printf("[dev] threads/SM           = %d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    printf("[dev] smem/block (opt-in)  = %d B (%.1f KB)\n", v, v / 1024.0);
    cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0);
    printf("[dev] SM count             = %d\n\n", v);

    size_t bA = (size_t)M * K * 4, bC = (size_t)M * N * 4;
    float *dA, *dB, *dC, *dR;
    cudaMalloc(&dA, bA); cudaMalloc(&dB, bA);
    cudaMalloc(&dC, bC); cudaMalloc(&dR, bC);
    {
        float* h = (float*)malloc(bA);
        for (size_t i = 0; i < (size_t)M * K; i++) h[i] = (float)(i % 7) * 0.1f;
        cudaMemcpy(dA, h, bA, cudaMemcpyHostToDevice);
        for (size_t i = 0; i < (size_t)M * K; i++) h[i] = (float)(i % 5) * 0.2f;
        cudaMemcpy(dB, h, bA, cudaMemcpyHostToDevice);
        free(h);
    }
    // 正确性：用 4/3 配置对 naive 校验一次（小尺寸省时间）
    {
        const int s2 = (SZ > 256) ? 256 : SZ;
        naive<<<512, 256>>>(dA, dB, dR, s2, s2, s2);
        cudaDeviceSynchronize();
        double t = bench<128, 128, 16, 4, 3, 0>(dA, dB, dC, s2, s2, s2);
        cudaDeviceSynchronize();
        double *a, *b; cudaMalloc(&a, 8); cudaMalloc(&b, 8);
        cudaMemset(a, 0, 8); cudaMemset(b, 0, 8);
        errk<<<1024, 256>>>(dC, dR, (long long)s2 * s2, a, b);
        double ha = 0, hb = 0;
        cudaMemcpy(&ha, a, 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(&hb, b, 8, cudaMemcpyDeviceToHost);
        printf("正确性校验(N=%d): rel_err=%.2e  %s  (该配置 %.1f us)\n\n", s2,
               (hb > 0) ? sqrt(ha / hb) : 0.0,
               ((hb > 0) && sqrt(ha / hb) < 1e-3) ? "PASS" : "FAIL", t);
        cudaFree(a); cudaFree(b);
    }

    const double gflop = 2.0 * (double)M * N * K / 1e12;
    printf("=== bufprobe: 128x128/TK16, N=%d, grid=%d, block=256 ===\n", SZ, NUM_SMS);
    printf("%-4s %-4s %-5s %-6s %-9s %-9s %-8s %s\n",
           "B", "D", "PAD", "smemKB", "us", "TFLOPS", "vs基准", "说明");
    double base = 0;
    for (int i = 0; i < (int)(sizeof(CASES) / sizeof(CASES[0])); i++) {
        if (CASES[i].B == 1) continue;
        double us = -3;
        switch (i) {
            case 0:  us = bench<128, 128, 16, 2, 1, 0>(dA, dB, dC, M, N, K); break;
            case 1:  us = bench<128, 128, 16, 3, 2, 0>(dA, dB, dC, M, N, K); break;
            case 2:  us = bench<128, 128, 16, 4, 3, 0>(dA, dB, dC, M, N, K); break;
            case 3:  us = bench<128, 128, 16, 5, 4, 0>(dA, dB, dC, M, N, K); break;
            case 4:  us = bench<128, 128, 16, 3, 1, 0>(dA, dB, dC, M, N, K); break;
            case 5:  us = bench<128, 128, 16, 4, 1, 0>(dA, dB, dC, M, N, K); break;
            case 6:  us = bench<128, 128, 16, 4, 2, 0>(dA, dB, dC, M, N, K); break;
            case 7:  us = bench<128, 128, 16, 5, 2, 0>(dA, dB, dC, M, N, K); break;
            case 8:  us = bench<128, 128, 16, 6, 2, 0>(dA, dB, dC, M, N, K); break;
            case 9:  us = bench<128, 128, 16, 5, 3, 0>(dA, dB, dC, M, N, K); break;
            case 10: us = bench<128, 128, 16, 6, 3, 0>(dA, dB, dC, M, N, K); break;
            case 11: us = bench<128, 128, 16, 6, 4, 0>(dA, dB, dC, M, N, K); break;
            case 12: us = bench<128, 128, 16, 4, 3, 32>(dA, dB, dC, M, N, K); break;
            case 13: us = bench<128, 128, 16, 4, 3, 64>(dA, dB, dC, M, N, K); break;
            case 14: us = bench<128, 128, 16, 5, 4, 32>(dA, dB, dC, M, N, K); break;
            case 15: us = bench<128, 128, 16, 5, 4, 64>(dA, dB, dC, M, N, K); break;
            default: us = -3; break;
        }
        const int smemKB = (CASES[i].B * (128 * 16 + CASES[i].PAD) +
                            CASES[i].B * (16 * 128 + CASES[i].PAD)) * 4 / 1024;
        if (us <= 0) { printf("%-4d %-4d %-5d %-6d %-9s %s\n",
                              CASES[i].B, CASES[i].D, CASES[i].PAD, smemKB,
                              "skip", CASES[i].note); continue; }
        if (base == 0) base = us;
        printf("%-4d %-4d %-5d %-6d %-9.1f %-9.2f %+7.1f%% %s\n",
               CASES[i].B, CASES[i].D, CASES[i].PAD, smemKB, us,
               gflop / (us * 1e-6), (us / base - 1) * 100, CASES[i].note);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dR);
    return 0;
}
