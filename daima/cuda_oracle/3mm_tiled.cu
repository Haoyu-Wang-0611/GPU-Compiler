// ============================================================
// 3mm_tiled.cu — 3mm 组化 pass-chain oracle（三遍 SGEMM 串联扫参）
//
// 结构对应 three_mm_probe.mlir（v1.8 组化分支决策层的穷举背书载具）：
//   pass1: E = A·B    pass2: F = C·D    pass3: G = E·F （三遍同尺寸 512³ SGEMM）
//   每 config：三遍共用同一 (BM, BN, TK, S)——组化模型"全 pass 参数共享"语义
//   每 iter 计时 = 三次 launch 串行（同流，无跨 pass 重叠）
//
// tiled kernel 与 sgemm_tiled.cu 逐字节同源；smem 口径 = (S+1)×(BM×TK + TK×BN)×4
//
// 【预注册（11-13 动码前）】
//   链最优 ≈ 81–84 µs（预期参数 (64,64,64,2)，同 twomm 链最优参数 ×3 passes）；
//   模型决策 (64,64,32,4)：predicted 1.250799e+05 ns（= 3×gemm512 单 pass 预测
//   + 1×launch），实测预计 ≈ 88–92 µs，rank ~3/48、距优 ~+9–12% → 双达标预期。
//
// 用法: ./3mm_tiled [SIZE] [BM BN TK S]
//   无参        → 512 全扫参
//   1 个参数    → 指定尺寸全扫参
//   4 个参数    → 512 + 标注模型决策
//   5 个参数    → 指定尺寸 + 标注模型决策
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define NUM_SMS 128
#define SMEM_CAP (99 * 1024)

// ---------------- 分块 + 流水 kernel（与 sgemm_tiled.cu 逐字节同源） ----------------
// A: M×K 行主序, B: K×N 行主序, C: M×N 行主序
// 约束（host 侧保证）：非整除时 ceil 网格 + 逐元素零填充/写守卫
template <int BM, int BN, int TK, int S>
__global__ void sgemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
    constexpr int TM = 4, TN = 4;
    constexpr int TP = (BM / TM) * (BN / TN) / 256;
    static_assert(BM % TM == 0 && BN % TN == 0, "tile not divisible");
    static_assert((BM / TM) * (BN / TN) % 256 == 0, "threads not covered");

    extern __shared__ float smem[];
    float* smA = smem;
    float* smB = smem + (S + 1) * BM * TK;

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

        for (int s = 0; s < S; s++) {
            if (s < nchunks) {
                float* sA = smA + s * BM * TK;
                float* sB = smB + s * TK * BN;
                for (int i = tid; i < BM * TK; i += 256) {
                    int r = i / TK, cc = i % TK;
                    int gr = bm0 + r, gc = s * TK + cc;
                    const float* src = A + (size_t)min(gr, M - 1) * K + min(gc, K - 1);
                    if (gr < M && gc < K)
                        __pipeline_memcpy_async(sA + i, src, sizeof(float));
                    else
                        sA[i] = 0.f;
                }
                for (int i = tid; i < TK * BN; i += 256) {
                    int r = i / BN, cc = i % BN;
                    int gr = s * TK + r, gc = bn0 + cc;
                    const float* src = B + (size_t)min(gr, K - 1) * N + min(gc, N - 1);
                    if (gr < K && gc < N)
                        __pipeline_memcpy_async(sB + i, src, sizeof(float));
                    else
                        sB[i] = 0.f;
                }
            }
            __pipeline_commit();
        }

        for (int kc = 0; kc < nchunks; kc++) {
            const int slot = kc % (S + 1);
            __pipeline_wait_prior(S - 1);
            __syncthreads();

            const float* sA = smA + slot * BM * TK;
            const float* sB = smB + slot * TK * BN;

#pragma unroll
            for (int p = 0; p < TP; p++) {
                const int gidx = p * 256 + tid;
                const int trow = (gidx / (BN / TN)) * TM;
                const int tcol = (gidx % (BN / TN)) * TN;
                for (int k = 0; k < TK; k++) {
                    const float a0 = sA[(trow + 0) * TK + k];
                    const float a1 = sA[(trow + 1) * TK + k];
                    const float a2 = sA[(trow + 2) * TK + k];
                    const float a3 = sA[(trow + 3) * TK + k];
                    const float b0 = sB[k * BN + tcol + 0];
                    const float b1 = sB[k * BN + tcol + 1];
                    const float b2 = sB[k * BN + tcol + 2];
                    const float b3 = sB[k * BN + tcol + 3];
                    acc[p][0][0] += a0 * b0; acc[p][0][1] += a0 * b1; acc[p][0][2] += a0 * b2; acc[p][0][3] += a0 * b3;
                    acc[p][1][0] += a1 * b0; acc[p][1][1] += a1 * b1; acc[p][1][2] += a1 * b2; acc[p][1][3] += a1 * b3;
                    acc[p][2][0] += a2 * b0; acc[p][2][1] += a2 * b1; acc[p][2][2] += a2 * b2; acc[p][2][3] += a2 * b3;
                    acc[p][3][0] += a3 * b0; acc[p][3][1] += a3 * b1; acc[p][3][2] += a3 * b2; acc[p][3][3] += a3 * b3;
                }
            }
            __syncthreads();

            const int nk = kc + S;
            if (nk < nchunks) {
                const int ns = nk % (S + 1);
                float* sA = smA + ns * BM * TK;
                float* sB = smB + ns * TK * BN;
                for (int i = tid; i < BM * TK; i += 256) {
                    int r = i / TK, cc = i % TK;
                    int gr = bm0 + r, gc = nk * TK + cc;
                    const float* src = A + (size_t)min(gr, M - 1) * K + min(gc, K - 1);
                    if (gr < M && gc < K)
                        __pipeline_memcpy_async(sA + i, src, sizeof(float));
                    else
                        sA[i] = 0.f;
                }
                for (int i = tid; i < TK * BN; i += 256) {
                    int r = i / BN, cc = i % BN;
                    int gr = nk * TK + r, gc = bn0 + cc;
                    const float* src = B + (size_t)min(gr, K - 1) * N + min(gc, N - 1);
                    if (gr < K && gc < N)
                        __pipeline_memcpy_async(sB + i, src, sizeof(float));
                    else
                        sB[i] = 0.f;
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

// ---------------- naive 参考 kernel（正确性基准，三遍链式调用） ----------------
__global__ void sgemm_naive(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)M * N;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < total; i += stride) {
        int r = (int)(i / N), c = (int)(i % N);
        float s = 0.f;
        const float* a = A + (size_t)r * K;
        const float* b = B + c;
        for (int k = 0; k < K; k++) s += a[k] * b[(size_t)k * N];
        C[i] = s;
    }
}

__global__ void err_kernel(const float* C, const float* Ref, long long total,
                           double* sum2, double* ref2) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < total; i += stride) {
        double d = (double)C[i] - (double)Ref[i];
        atomicAdd(sum2, d * d);
        atomicAdd(ref2, (double)Ref[i] * Ref[i]);
    }
}

// ---------------- 计时封装：每 iter = pass1(E=A·B) + pass2(F=C·D) + pass3(G=E·F) ----------------
template <int BM, int BN, int TK, int S>
double bench3(const float* dA, const float* dB, const float* dC, const float* dD,
              float* dE, float* dF, float* dG, int M, int N, int K) {
    const int smem = (S + 1) * (BM * TK + TK * BN) * 4;
    static bool attr_done = false;
    if (!attr_done) {
        cudaError_t e = cudaFuncSetAttribute(sgemm_tiled<BM, BN, TK, S>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_CAP);
        if (e != cudaSuccess) {
            printf("  [warn] cudaFuncSetAttribute failed (%dx%d/%d/%d)\n", BM, BN, TK, S);
            return -1.0;
        }
        attr_done = true;
    }
    dim3 grid(NUM_SMS), block(256);
    for (int w = 0; w < 3; w++) {
        sgemm_tiled<BM, BN, TK, S><<<grid, block, smem>>>(dA, dB, dE, M, N, K);
        sgemm_tiled<BM, BN, TK, S><<<grid, block, smem>>>(dC, dD, dF, M, N, K);
        sgemm_tiled<BM, BN, TK, S><<<grid, block, smem>>>(dE, dF, dG, M, N, K);
    }
    cudaDeviceSynchronize();
    {
        cudaError_t e2 = cudaGetLastError();
        if (e2 != cudaSuccess) {
            printf("  [fail] launch error (%dx%d/%d/%d): %s\n", BM, BN, TK, S,
                   cudaGetErrorString(e2));
            return -1.0;
        }
    }

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int it = 0; it < 20; it++) {
        sgemm_tiled<BM, BN, TK, S><<<grid, block, smem>>>(dA, dB, dE, M, N, K);
        sgemm_tiled<BM, BN, TK, S><<<grid, block, smem>>>(dC, dD, dF, M, N, K);
        sgemm_tiled<BM, BN, TK, S><<<grid, block, smem>>>(dE, dF, dG, M, N, K);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (double(ms) / 20.0) * 1000.0;  // us（三遍合计）
}

// ---------------- dispatch 宏（8 对 BM×BN × 3 TK × 4 S） ----------------
#define DISPATCH_S(BM, BN, TK)                                                            \
    if (S == 1) us = bench3<BM, BN, TK, 1>(dA, dB, dC, dD, dE, dF, dG, M, N, K);          \
    else if (S == 2) us = bench3<BM, BN, TK, 2>(dA, dB, dC, dD, dE, dF, dG, M, N, K);     \
    else if (S == 3) us = bench3<BM, BN, TK, 3>(dA, dB, dC, dD, dE, dF, dG, M, N, K);     \
    else if (S == 4) us = bench3<BM, BN, TK, 4>(dA, dB, dC, dD, dE, dF, dG, M, N, K);

#define DISPATCH_TK(BM, BN)                       \
    if (TK == 16) { DISPATCH_S(BM, BN, 16) }      \
    else if (TK == 32) { DISPATCH_S(BM, BN, 32) } \
    else if (TK == 64) { DISPATCH_S(BM, BN, 64) }

#define DISPATCH_BN(BM)                            \
    if (BN == 64) { DISPATCH_TK(BM, 64) }          \
    else if (BN == 128) { DISPATCH_TK(BM, 128) }   \
    else if (BN == 256) { DISPATCH_TK(BM, 256) }

// ---------------- 排名表条目 ----------------
struct Row {
    int BM, BN, TK, S;
    double us;
    double smemKB;
};

int cmpRow(const void* a, const void* b) {
    const Row* x = (const Row*)a;
    const Row* y = (const Row*)b;
    return (x->us > y->us) - (x->us < y->us);
}

int main(int argc, char** argv) {
    // 按"参数个数"解析（nv 不含程序名——11-08 修正，与 sgemm_tiled 同款）
    int SZ = 512, mBM = 0, mBN = 0, mTK = 0, mS = 0;
    int vals[5] = {0, 0, 0, 0, 0}, nv = 0;
    for (int i = 1; i < argc && nv < 5; ++i) vals[nv++] = atoi(argv[i]);
    if (nv == 1) { SZ = vals[0]; }
    else if (nv == 4) { mBM = vals[0]; mBN = vals[1]; mTK = vals[2]; mS = vals[3]; }
    else if (nv == 5) { SZ = vals[0]; mBM = vals[1]; mBN = vals[2]; mTK = vals[3]; mS = vals[4]; }
    else { printf("[参数] 用法: ./3mm_tiled [SIZE] [BM BN TK S]\n"); }
    printf("[参数] SZ=%d, 模型标注: %d,%d,%d,%d\n", SZ, mBM, mBN, mTK, mS);
    const int M = SZ, N = SZ, K = SZ;
    const size_t bytesM = (size_t)M * N * 4, bytesK = (size_t)M * K * 4;
    float *dA, *dB, *dC, *dD, *dE, *dF, *dG, *dERef, *dFRef, *dGRef;
    cudaMalloc(&dA, bytesK); cudaMalloc(&dB, bytesK);
    cudaMalloc(&dC, bytesK); cudaMalloc(&dD, bytesK);
    cudaMalloc(&dE, bytesM); cudaMalloc(&dF, bytesM); cudaMalloc(&dG, bytesM);
    cudaMalloc(&dERef, bytesM); cudaMalloc(&dFRef, bytesM); cudaMalloc(&dGRef, bytesM);

    // 初始化（简单可复现的值，与 sgemm_tiled 同款）
    float* hA = (float*)malloc(bytesK);
    for (size_t i = 0; i < M * K; i++) hA[i] = (float)(i % 7) * 0.1f;
    cudaMemcpy(dA, hA, bytesK, cudaMemcpyHostToDevice);
    float* hB = (float*)malloc(bytesK);
    for (size_t i = 0; i < K * N; i++) hB[i] = (float)(i % 5) * 0.2f;
    cudaMemcpy(dB, hB, bytesK, cudaMemcpyHostToDevice);
    float* hC = (float*)malloc(bytesK);
    for (size_t i = 0; i < M * N; i++) hC[i] = (float)(i % 3) * 0.05f;
    cudaMemcpy(dC, hC, bytesK, cudaMemcpyHostToDevice);
    float* hD = (float*)malloc(bytesK);
    for (size_t i = 0; i < M * N; i++) hD[i] = (float)(i % 11) * 0.02f;
    cudaMemcpy(dD, hD, bytesK, cudaMemcpyHostToDevice);
    free(hA); free(hB); free(hC); free(hD);

    printf("=== 3mm 组化链 oracle: pass1 E=A·B + pass2 F=C·D + pass3 G=E·F (M=N=K=%d, FP32, 三遍共用配置) ===\n", SZ);
    printf("  候选空间: BM/BN∈{64,128,256}², TK∈{16,32,64}, S∈{1..4}, smem≤99KB（三遍同配置）\n");
    printf("  排除: 256x256（每线程 256 累加寄存器 > 255 硬上限，10-31 实证）\n\n");

    // ---- 正确性校验：naive 三遍链 vs tiled 三遍链，对最终 G 判误差 ----
    {
        int threads = 256, blocks = (M * N / 256);
        sgemm_naive<<<blocks, threads>>>(dA, dB, dERef, M, N, K);
        sgemm_naive<<<blocks, threads>>>(dC, dD, dFRef, M, N, K);
        sgemm_naive<<<blocks, threads>>>(dERef, dFRef, dGRef, M, N, K);
        cudaDeviceSynchronize();
        sgemm_tiled<64, 64, 16, 1><<<NUM_SMS, 256, (2) * (64 * 16 + 16 * 64) * 4>>>(dA, dB, dE, M, N, K);
        sgemm_tiled<64, 64, 16, 1><<<NUM_SMS, 256, (2) * (64 * 16 + 16 * 64) * 4>>>(dC, dD, dF, M, N, K);
        sgemm_tiled<64, 64, 16, 1><<<NUM_SMS, 256, (2) * (64 * 16 + 16 * 64) * 4>>>(dE, dF, dG, M, N, K);
        cudaDeviceSynchronize();

        double *d_sum2, *d_ref2, h_sum2 = 0, h_ref2 = 0;
        cudaMalloc(&d_sum2, 8); cudaMalloc(&d_ref2, 8);
        cudaMemset(d_sum2, 0, 8); cudaMemset(d_ref2, 0, 8);
        err_kernel<<<1024, 256>>>(dG, dGRef, (long long)M * N, d_sum2, d_ref2);
        cudaMemcpy(&h_sum2, d_sum2, 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_ref2, d_ref2, 8, cudaMemcpyDeviceToHost);
        double rel = (h_ref2 > 0) ? __builtin_sqrt(h_sum2 / h_ref2) : 0;
        printf("正确性校验(三遍链, 对 G): %s (rel_err = %.2e)\n\n", rel < 1e-3 ? "PASS" : "FAIL", rel);
        cudaFree(d_sum2); cudaFree(d_ref2);
    }

    // ---- 扫参（候选空间与 smem 口径 = L1Feas 组化分支，三遍同配置）----
    const int BMs[3] = {64, 128, 256};
    const int BNs[3] = {64, 128, 256};
    const int TKs[3] = {16, 32, 64};
    const int Ss[4] = {1, 2, 3, 4};
    const double gflop = 2.0 * 3.0 * M * N * K / 1e12;  // 三遍

    Row rows[48];
    int nRows = 0;
    double bestUs = 1e18; int bBM = 0, bBN = 0, bTK = 0, bS = 0;

    for (int bi = 0; bi < 3; bi++) {
        for (int bj = 0; bj < 3; bj++) {
            int BM = BMs[bi], BN = BNs[bj];
            if (BM == 256 && BN == 256) continue;
            printf("--- BM=%d BN=%d ---\n", BM, BN);
            for (int ti = 0; ti < 3; ti++) {
                for (int si = 0; si < 4; si++) {
                    int TK = TKs[ti], S = Ss[si];
                    long long smem = (long long)(S + 1) * (BM * TK + TK * BN) * 4;
                    if (smem > SMEM_CAP) continue;
                    double us = -1;
                    if (BM == 64) { DISPATCH_BN(64) }
                    else if (BM == 128) { DISPATCH_BN(128) }
                    else if (BM == 256) {
                        if (BN == 64) { DISPATCH_TK(256, 64) }
                        else if (BN == 128) { DISPATCH_TK(256, 128) }
                    }
                    if (us < 0) continue;
                    double tflops = gflop / (us * 1e-6);
                    printf("  BM=%-4d BN=%-4d TK=%-3d S=%d  smem=%5.1fKB  耗时: %9.1f us  (%5.1f TFLOPS)\n",
                           BM, BN, TK, S, smem / 1024.0, us, tflops);
                    rows[nRows++] = {BM, BN, TK, S, us, smem / 1024.0};
                    if (us < bestUs) { bestUs = us; bBM = BM; bBN = BN; bTK = TK; bS = S; }
                }
            }
        }
    }

    printf("\n=== 最优: BM=%d BN=%d TK=%d S=%d @ %.1f us (%.1f TFLOPS, 三遍合计) ===\n",
           bBM, bBN, bTK, bS, bestUs, gflop / (bestUs * 1e-6));

    // ---- 实测排名 TOP-20 ----
    qsort(rows, nRows, sizeof(Row), cmpRow);
    int top = nRows < 20 ? nRows : 20;
    printf("\n===== 实测排名 TOP-%d（共 %d 配置，三遍合计）=====\n", top, nRows);
    printf("%-5s %-6s %-6s %-5s %-3s %10s\n", "rank", "BM", "BN", "TK", "S", "耗时(us)");
    for (int i = 0; i < top; i++) {
        printf("%-5d %-6d %-6d %-5d %-3d %10.1f",
               i + 1, rows[i].BM, rows[i].BN, rows[i].TK, rows[i].S, rows[i].us);
        if (mBM && rows[i].BM == mBM && rows[i].BN == mBN &&
            rows[i].TK == mTK && rows[i].S == mS)
            printf("  <<<< 模型决策");
        printf("\n");
    }
    if (mBM) {
        int rank = -1;
        for (int i = 0; i < nRows; i++)
            if (rows[i].BM == mBM && rows[i].BN == mBN && rows[i].TK == mTK && rows[i].S == mS)
                { rank = i + 1; break; }
        if (rank > 0)
            printf("[判定] 模型决策 (%d,%d,%d,%d): 穷举第 %d/%d 名 = 前 %.1f%%，距最优 %+.1f%% → %s\n",
                   mBM, mBN, mTK, mS, rank, nRows, rank * 100.0 / nRows,
                   (rows[rank - 1].us / rows[0].us - 1) * 100.0,
                   (rows[rank - 1].us / rows[0].us <= 1.20 && rank * 100.0 / nRows <= 20.0)
                       ? "前20% 达标" : "未达标");
        else
            printf("[判定] 模型决策 (%d,%d,%d,%d) 不在实测集中\n", mBM, mBN, mTK, mS);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    cudaFree(dE); cudaFree(dF); cudaFree(dG);
    cudaFree(dERef); cudaFree(dFRef); cudaFree(dGRef);
    return 0;
}
