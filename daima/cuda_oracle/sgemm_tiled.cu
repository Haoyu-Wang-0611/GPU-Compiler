// ============================================================
// sgemm_tiled.cu — 分块 SGEMM + cp.async 多级流水扫参
//
// 结构对应 l2_gemm.py 的物理语义：
//   grid = NUM_SMS(128)，每 block 按波次扫多个输出 tile（= 模型 waves）
//   每 tile 内沿 K 维按 TK 切 chunk，(S+1) 份 smem 缓冲，cp.async 流水
//   smem 分配 = (S+1)×(BM×TK + TK×BN)×4 —— 与 L1Feas GEMM 模板逐字节同口径
//
// 256x256 被排除：每线程需 256 个 FP32 累加寄存器 > 255 硬上限
//   （10-31 实证，L1Feas GEMM 模板同款约束）
//
// 用法: ./sgemm_tiled [模型BM 模型BN 模型TK 模型S]
//   传入模型决策四元组时，TOP-20 表中对应行会标注"←模型决策"
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define NUM_SMS 128
#define SMEM_CAP (99 * 1024)

// ---------------- 分块 + 流水 kernel ----------------
// A: M×K 行主序, B: K×N 行主序, C: M×N 行主序
// 约束（host 侧保证）：M%BM==0, N%BN==0, K%TK==0
template <int BM, int BN, int TK, int S>
__global__ void sgemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
    constexpr int TM = 4, TN = 4;
    constexpr int TP = (BM / TM) * (BN / TN) / 256;  // 每线程子块遍数
    static_assert(BM % TM == 0 && BN % TN == 0, "tile not divisible");
    static_assert((BM / TM) * (BN / TN) % 256 == 0, "threads not covered");

    extern __shared__ float smem[];
    float* smA = smem;                      // (S+1) 份 BM×TK
    float* smB = smem + (S + 1) * BM * TK;  // (S+1) 份 TK×BN

    const int G = gridDim.x;
    // 11-08 边界守卫版：ceil 网格 + 逐元素零填充/写守卫（支持非整除尺寸，gemm500 oracle）
    const int tilesN = (N + BN - 1) / BN;
    const int total_tiles = ((M + BM - 1) / BM) * tilesN;
    const int nchunks = (K + TK - 1) / TK;
    const int tid = threadIdx.x;

    // 累加器：跨 K 循环存活（每线程 TP 个 4×4 子块）
    float acc[TP][TM][TN];
#pragma unroll
    for (int p = 0; p < TP; p++)
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) acc[p][i][j] = 0.f;

    // 波次扫输出 tile（对应模型 waves 语义）
    const int my_tiles = (total_tiles - blockIdx.x + G - 1) / G;

    for (int c = 0; c < my_tiles; c++) {
        const int t = blockIdx.x + c * G;
        const int bm0 = (t / tilesN) * BM;
        const int bn0 = (t % tilesN) * BN;

        // ---- prologue：预取本 tile 的前 S 个 K-chunk ----
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
                        sA[i] = 0.f;   // OOB: cp.async 4B 不支持零填充（静默回退全量拷贝）→ 普通写零
                }
                for (int i = tid; i < TK * BN; i += 256) {
                    int r = i / BN, cc = i % BN;
                    int gr = s * TK + r, gc = bn0 + cc;
                    const float* src = B + (size_t)min(gr, K - 1) * N + min(gc, N - 1);
                    if (gr < K && gc < N)
                        __pipeline_memcpy_async(sB + i, src, sizeof(float));
                    else
                        sB[i] = 0.f;   // OOB: 同上，普通写零
                }
            }
            __pipeline_commit();
        }

        // ---- 主循环：算 chunk kc，预取 chunk kc+S ----
        for (int kc = 0; kc < nchunks; kc++) {
            const int slot = kc % (S + 1);
            __pipeline_wait_prior(S - 1);
            __syncthreads();

            const float* sA = smA + slot * BM * TK;
            const float* sB = smB + slot * TK * BN;

            // p 必须完全展开：acc[p] 才能驻留寄存器（运行时索引会落 local memory）
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

            // 预取未来 chunk 到腾出的槽位
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
                        sA[i] = 0.f;   // OOB: 普通写零
                }
                for (int i = tid; i < TK * BN; i += 256) {
                    int r = i / BN, cc = i % BN;
                    int gr = nk * TK + r, gc = bn0 + cc;
                    const float* src = B + (size_t)min(gr, K - 1) * N + min(gc, N - 1);
                    if (gr < K && gc < N)
                        __pipeline_memcpy_async(sB + i, src, sizeof(float));
                    else
                        sB[i] = 0.f;   // OOB: 同上，普通写零
                }
            }
            __pipeline_commit();
        }

        // ---- 收尾：排空在途 cp.async（防跨 tile 的 smem 写竞争）----
        __pipeline_wait_prior(0);
        __syncthreads();

        // ---- 写回 + 清累加器 ----
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

// ---------------- naive 参考 kernel（正确性基准） ----------------
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

// 误差统计：rel = sqrt(Σ(C-Ref)² / ΣRef²)
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

// ---------------- 计时封装 ----------------
template <int BM, int BN, int TK, int S>
double bench(const float* dA, const float* dB, float* dC, int M, int N, int K) {
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
    for (int w = 0; w < 3; w++)
        sgemm_tiled<BM, BN, TK, S><<<grid, block, smem>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    // 【保险】launch 失败必须拦截，否则空转计时会产生垃圾最优值
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
    for (int it = 0; it < 20; it++)
        sgemm_tiled<BM, BN, TK, S><<<grid, block, smem>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (double(ms) / 20.0) * 1000.0;  // us
}

// ---------------- dispatch 宏（8 对 BM×BN × 3 TK × 4 S） ----------------
#define DISPATCH_S(BM, BN, TK)                                                    \
    if (S == 1) us = bench<BM, BN, TK, 1>(dA, dB, dC, M, N, K);                   \
    else if (S == 2) us = bench<BM, BN, TK, 2>(dA, dB, dC, M, N, K);              \
    else if (S == 3) us = bench<BM, BN, TK, 3>(dA, dB, dC, M, N, K);              \
    else if (S == 4) us = bench<BM, BN, TK, 4>(dA, dB, dC, M, N, K);

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
    // 用法: ./sgemm_tiled [SIZE] [BM BN TK S]
    //   无参        → 4096 全扫参
    //   1 个参数    → 指定尺寸全扫参（如 500）
    //   4 个参数    → 4096 + 标注模型决策
    //   5 个参数    → 指定尺寸 + 标注模型决策
    // 按"参数个数"解析（nv 不含程序名，避免 argc 差一错误——11-08 修正）
    int SZ = 4096, mBM = 0, mBN = 0, mTK = 0, mS = 0;
    int vals[5] = {0, 0, 0, 0, 0}, nv = 0;
    for (int i = 1; i < argc && nv < 5; ++i) vals[nv++] = atoi(argv[i]);
    if (nv == 1) { SZ = vals[0]; }
    else if (nv == 4) { mBM = vals[0]; mBN = vals[1]; mTK = vals[2]; mS = vals[3]; }
    else if (nv == 5) { SZ = vals[0]; mBM = vals[1]; mBN = vals[2]; mTK = vals[3]; mS = vals[4]; }
    else { printf("[参数] 用法: ./sgemm_tiled [SIZE] [BM BN TK S]\n"); }
    printf("[参数] SZ=%d, 模型标注: %d,%d,%d,%d\n", SZ, mBM, mBN, mTK, mS);
    const int M = SZ, N = SZ, K = SZ;
    const size_t bytesA = (size_t)M * K * 4, bytesB = (size_t)K * N * 4, bytesC = (size_t)M * N * 4;
    float *dA, *dB, *dC, *dRef;
    cudaMalloc(&dA, bytesA); cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC); cudaMalloc(&dRef, bytesC);


    // 初始化（简单可复现的值）
    float* hA = (float*)malloc(bytesA);
    for (size_t i = 0; i < M * K; i++) hA[i] = (float)(i % 7) * 0.1f;
    cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice);
    float* hB = (float*)malloc(bytesB);
    for (size_t i = 0; i < K * N; i++) hB[i] = (float)(i % 5) * 0.2f;
    cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice);
    free(hA); free(hB);

    printf("=== GEMM tiled SGEMM 扫参 (M=N=K=%d, FP32, 128 blocks 波次语义) ===\n", SZ);
    printf("  候选空间: BM/BN∈{64,128,256}², TK∈{16,32,64}, S∈{1..4}, smem≤99KB\n");
    printf("  排除: 256x256（每线程 256 累加寄存器 > 255 硬上限，10-31 实证）\n\n");

    // ---- 正确性校验（对第一组有效配置；所有配置累加顺序相同，结果一致）----
    {
        int threads = 256, blocks = (M * N / 256);
        sgemm_naive<<<blocks, threads>>>(dA, dB, dRef, M, N, K);
        cudaDeviceSynchronize();
        sgemm_tiled<64, 64, 16, 1><<<NUM_SMS, 256, (2) * (64 * 16 + 16 * 64) * 4>>>(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();

        double *d_sum2, *d_ref2, h_sum2 = 0, h_ref2 = 0;
        cudaMalloc(&d_sum2, 8); cudaMalloc(&d_ref2, 8);
        cudaMemset(d_sum2, 0, 8); cudaMemset(d_ref2, 0, 8);
        err_kernel<<<1024, 256>>>(dC, dRef, (long long)M * N, d_sum2, d_ref2);
        cudaMemcpy(&h_sum2, d_sum2, 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_ref2, d_ref2, 8, cudaMemcpyDeviceToHost);
        double rel = (h_ref2 > 0) ? __builtin_sqrt(h_sum2 / h_ref2) : 0;
        printf("正确性校验: %s (rel_err = %.2e)\n\n", rel < 1e-3 ? "PASS" : "FAIL", rel);
        cudaFree(d_sum2); cudaFree(d_ref2);
    }

    // ---- 扫参（候选空间 = L1Feas GEMM 模板的枚举空间，smem 同口径）----
    const int BMs[3] = {64, 128, 256};
    const int BNs[3] = {64, 128, 256};
    const int TKs[3] = {16, 32, 64};
    const int Ss[4] = {1, 2, 3, 4};
    const double gflop = 2.0 * M * N * K / 1e12;

    Row rows[48];
    int nRows = 0;
    double bestUs = 1e18; int bBM = 0, bBN = 0, bTK = 0, bS = 0;

    for (int bi = 0; bi < 3; bi++) {
        for (int bj = 0; bj < 3; bj++) {
            int BM = BMs[bi], BN = BNs[bj];
            if (BM == 256 && BN == 256) continue;  // 寄存器约束排除
            printf("--- BM=%d BN=%d ---\n", BM, BN);
            for (int ti = 0; ti < 3; ti++) {
                for (int si = 0; si < 4; si++) {
                    int TK = TKs[ti], S = Ss[si];
                    long long smem = (long long)(S + 1) * (BM * TK + TK * BN) * 4;
                    if (smem > SMEM_CAP) continue;   // smem 红线（与模型同口径）
                    double us = -1;
                    // dispatch 链必须传字面量（模板实参要求编译期常量），
                    // 且 256x256 在编译期就不实例化（与寄存器约束一致）
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

    printf("\n=== 最优: BM=%d BN=%d TK=%d S=%d @ %.1f us (%.1f TFLOPS) ===\n",
           bBM, bBN, bTK, bS, bestUs, gflop / (bestUs * 1e-6));

    // ---- 实测排名 TOP-20（按耗时升序）----
    qsort(rows, nRows, sizeof(Row), cmpRow);
    int top = nRows < 20 ? nRows : 20;
    printf("\n===== 实测排名 TOP-%d（共 %d 配置）=====\n", top, nRows);
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
        // 模型决策的穷举名次与距最优比
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

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    return 0;
}
