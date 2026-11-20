// ============================================================
// stril_tiled.cu — symm/syrk 三角守卫形态 oracle（tile 级三角跳过）
//
// 载具语义 = 探针近似形态（与模型决策所依据的 IR 逐位对齐）：
//   MODE 0 (syrk): C[i,j] += A[i,k]·A[j,k]，守卫 k<j（列守卫）；
//                  A 双读：行带 (BM×TK, i-major) + 列带 (BN×TK, j-major 转置)
//   MODE 1 (symm): C[i,j] += A[k,i]·B[k,j]，守卫 k<i（行守卫）；
//                  A 面板 k-major (TK×BM)，B 面板 k-major (TK×BN)
//   两者均为 C 全矩阵 + k 三角守卫（探针无 j≤i 限制；官方下三角 tile 跳过
//   属 v1.9 登记项，不在本载具——工作负载必须与模型决策输入一致）
//
// 三角跳过的实现粒度（诚实载具，对应探针"load 在守卫内"语义）：
//   ① chunk 级：死 chunk（kc*TK ≥ 三角边界）不预取、不计算，空 commit
//      保持 (S+1) 流水记账对齐——流量与计算同时砍半；
//   ② 元素级：活 chunk 内按守卫条件分支无掩码乘（b×(gk<col) 或 a×(gk<row)），
//      保证与 naive 参考（同序累加）逐位一致。
//
// 【预注册（11-15 v2，模型 v1.8.3 双面板修正后）】
//   MODE 0 syrk: 模型决策 (64,64,TK=32,S=4)，predicted 4.169331e+04 ns
//     （双面板修正后 syrk 流量 = gemm 的 A+B 等量 → 决策回归 gemm 同参；
//      v1 预注册的 (64,64,64,4)/"TK=64 自适应"系合并窗口伪影，已撤回）
//   MODE 1 symm: 模型决策 (64,64,TK=32,S=4)，predicted 4.169331e+04 ns
//   首轮实测（11-15）：symm rank 3/48 = 6.2%、距优 +7.0% ✓；syrk（v2 载具）
//   (64,64,32,4) rank 7/48 = 14.6%、距优 +8.0% → 双达标。
//   守卫浪费不计入模型（满 trip 记账）→ 实测绝对值 ≈ predicted 的 0.6~0.85×，
//   实测/预测比 = A5 守卫浪费项的量化数据；
//   判定 = 全空间（48 配置）rank ≤20% 且距优 ≤20%（双达标）。
//   【噪声带登记】25–30 µs kernel 的 best-of-3 波动 ±5–8%（symm 两轮 rank 2↔3），
//   近并列按并列处理——oracle 交错重复测量协议（留出套件必须纳入）。
//
// 【保险继承】cudaFuncSetAttribute（>48KB smem）+ launch 失败拦截
//   （11-14 heat1d_oracle 教训：静默失败产生 2.2 µs 假数据）。
//
// 用法: ./stril_tiled <mode> [SIZE] [BM BN TK S]
//   mode: 0=syrk 1=symm；SIZE 默认 512；
//   模型标注缺省：syrk→64,64,32,4  symm→64,64,32,4（均为 v1.8.3 决策）
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define NUM_SMS 128
#define SMEM_CAP (99 * 1024)

// ---------------- 三角守卫分块 + 流水 kernel ----------------
// MODE 0: syrk  — A: N×K 行主序（双重读：i-major 行带 + j-major 列带）
// MODE 1: symm  — A: K×N（按 [k,i] 读 = k-major 面板）, B: K×N 行主序
// 约束（host 侧保证）：N % BM == 0, N % BN == 0, K % TK == 0（512 全整除）
template <int BM, int BN, int TK, int S, int MODE>
__global__ void stril_tiled(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int N, int K) {
    constexpr int TM = 4, TN = 4;
    constexpr int TP = (BM / TM) * (BN / TN) / 256;
    static_assert(BM % TM == 0 && BN % TN == 0, "tile not divisible");
    static_assert((BM / TM) * (BN / TN) % 256 == 0, "threads not covered");

    extern __shared__ float smem[];
    float* smA = smem;                       // MODE0: (S+1)×BM×TK | MODE1: (S+1)×TK×BM
    float* smB = smem + (S + 1) * BM * TK;   // MODE0: (S+1)×BN×TK | MODE1: (S+1)×TK×BN
    // 注：两模式 smem 总字节相同（BM*TK+TK*BN），仅解释/索引方式不同

    const int G = gridDim.x;
    const int tilesN = (N + BN - 1) / BN;
    const int total_tiles = ((N + BM - 1) / BM) * tilesN;
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

        // 三角边界（本 tile 的 k 上界）：syrk 列守卫 k<j → k < bn0+BN-1；
        // symm 行守卫 k<i → k < bm0+BM-1。chunk 活性 = kc*TK < 该边界。
        const int liveLimit = (MODE == 0) ? (bn0 + BN - 1) : (bm0 + BM - 1);

        // ---- prologue：预取前 S 个活 chunk（死 chunk 空 commit）----
        for (int s = 0; s < S; s++) {
            if (s < nchunks && s * TK < liveLimit) {
                if (MODE == 0) {
                    float* sA = smA + s * BM * TK;
                    for (int i = tid; i < BM * TK; i += 256) {
                        int r = i / TK, cc = i % TK;              // r=i 偏移, cc=k 偏移
                        int gi = bm0 + r, gk = s * TK + cc;
                        if (gi < N && gk < K)
                            __pipeline_memcpy_async(sA + i, A + (size_t)gi * K + gk, sizeof(float));
                        else
                            sA[i] = 0.f;
                    }
                    // 第二面板（A[j][k]）：全局按 kk fastest 读（行连续合并），
                    // smem 写 k-major（kk*BN+cc）→ 计算端 sB[k*BN+j] 无库冲突
                    //（v1.1 修正：原 j-major 布局计算端步长 4·TK ≡ 0 mod 32，
                    //  16~32-way 同库冲突，实测污染排名——11-15 首轮教训）
                    float* sB = smB + s * BN * TK;
                    for (int i = tid; i < BN * TK; i += 256) {
                        int cc = i / TK, kk = i % TK;             // cc=j 偏移, kk=k 偏移（kk fastest）
                        int gj = bn0 + cc, gk = s * TK + kk;
                        if (gj < N && gk < K)
                            __pipeline_memcpy_async(sB + kk * BN + cc,
                                                    A + (size_t)gj * K + gk, sizeof(float));
                        else
                            sB[kk * BN + cc] = 0.f;
                    }
                } else {
                    float* sA = smA + s * TK * BM;
                    for (int i = tid; i < TK * BM; i += 256) {
                        int r = i / BM, cc = i % BM;              // r=k 偏移, cc=i 偏移
                        int gk = s * TK + r, gi = bm0 + cc;
                        if (gk < K && gi < N)
                            __pipeline_memcpy_async(sA + i, A + (size_t)gk * K + gi, sizeof(float));
                        else
                            sA[i] = 0.f;
                    }
                    float* sB = smB + s * TK * BN;
                    for (int i = tid; i < TK * BN; i += 256) {
                        int r = i / BN, cc = i % BN;              // r=k 偏移, cc=j 偏移
                        int gk = s * TK + r, gj = bn0 + cc;
                        if (gk < K && gj < N)
                            __pipeline_memcpy_async(sB + i, B + (size_t)gk * N + gj, sizeof(float));
                        else
                            sB[i] = 0.f;
                    }
                }
            }
            __pipeline_commit();   // 死 chunk 也 commit（空批次）→ 记账与 wait_prior 对齐
        }

        // ---- 主循环：算活 chunk kc，预取 chunk kc+S ----
        for (int kc = 0; kc < nchunks; kc++) {
            const int slot = kc % (S + 1);
            __pipeline_wait_prior(S - 1);
            __syncthreads();

            const bool live = (kc * TK < liveLimit);
            if (live) {
                const float* sA = (MODE == 0) ? smA + slot * BM * TK : smA + slot * TK * BM;
                const float* sB = (MODE == 0) ? smB + slot * BN * TK : smB + slot * TK * BN;
#pragma unroll
                for (int p = 0; p < TP; p++) {
                    const int gidx = p * 256 + tid;
                    const int trow = (gidx / (BN / TN)) * TM;
                    const int tcol = (gidx % (BN / TN)) * TN;
                    for (int k = 0; k < TK; k++) {
                        const int gk = kc * TK + k;
                        float a0, a1, a2, a3, b0, b1, b2, b3;
                        if (MODE == 0) {
                            a0 = sA[(trow + 0) * TK + k];
                            a1 = sA[(trow + 1) * TK + k];
                            a2 = sA[(trow + 2) * TK + k];
                            a3 = sA[(trow + 3) * TK + k];
                            b0 = sB[k * BN + tcol + 0];     // A[j][k]：k-major smem（无库冲突）
                            b1 = sB[k * BN + tcol + 1];
                            b2 = sB[k * BN + tcol + 2];
                            b3 = sB[k * BN + tcol + 3];
                            // 列守卫 k<j：分支无掩码乘（×1.0 精确，×0.0 加零不改和）
                            b0 *= (gk < bn0 + tcol + 0) ? 1.f : 0.f;
                            b1 *= (gk < bn0 + tcol + 1) ? 1.f : 0.f;
                            b2 *= (gk < bn0 + tcol + 2) ? 1.f : 0.f;
                            b3 *= (gk < bn0 + tcol + 3) ? 1.f : 0.f;
                        } else {
                            a0 = sA[k * BM + trow + 0];     // A[k][i]：k-major
                            a1 = sA[k * BM + trow + 1];
                            a2 = sA[k * BM + trow + 2];
                            a3 = sA[k * BM + trow + 3];
                            b0 = sB[k * BN + tcol + 0];
                            b1 = sB[k * BN + tcol + 1];
                            b2 = sB[k * BN + tcol + 2];
                            b3 = sB[k * BN + tcol + 3];
                            // 行守卫 k<i：掩码乘在 a 侧
                            a0 *= (gk < bm0 + trow + 0) ? 1.f : 0.f;
                            a1 *= (gk < bm0 + trow + 1) ? 1.f : 0.f;
                            a2 *= (gk < bm0 + trow + 2) ? 1.f : 0.f;
                            a3 *= (gk < bm0 + trow + 3) ? 1.f : 0.f;
                        }
                        acc[p][0][0] += a0 * b0; acc[p][0][1] += a0 * b1; acc[p][0][2] += a0 * b2; acc[p][0][3] += a0 * b3;
                        acc[p][1][0] += a1 * b0; acc[p][1][1] += a1 * b1; acc[p][1][2] += a1 * b2; acc[p][1][3] += a1 * b3;
                        acc[p][2][0] += a2 * b0; acc[p][2][1] += a2 * b1; acc[p][2][2] += a2 * b2; acc[p][2][3] += a2 * b3;
                        acc[p][3][0] += a3 * b0; acc[p][3][1] += a3 * b1; acc[p][3][2] += a3 * b2; acc[p][3][3] += a3 * b3;
                    }
                }
            }
            __syncthreads();

            // 预取未来 chunk（仅活 chunk 发起传输；空 commit 对齐记账）
            const int nk = kc + S;
            if (nk < nchunks && nk * TK < liveLimit) {
                const int ns = nk % (S + 1);
                if (MODE == 0) {
                    float* sA = smA + ns * BM * TK;
                    for (int i = tid; i < BM * TK; i += 256) {
                        int r = i / TK, cc = i % TK;
                        int gi = bm0 + r, gk = nk * TK + cc;
                        if (gi < N && gk < K)
                            __pipeline_memcpy_async(sA + i, A + (size_t)gi * K + gk, sizeof(float));
                        else
                            sA[i] = 0.f;
                    }
                    float* sB = smB + ns * BN * TK;   // k-major（同 prologue，见上注）
                    for (int i = tid; i < BN * TK; i += 256) {
                        int cc = i / TK, kk = i % TK;
                        int gj = bn0 + cc, gk = nk * TK + kk;
                        if (gj < N && gk < K)
                            __pipeline_memcpy_async(sB + kk * BN + cc,
                                                    A + (size_t)gj * K + gk, sizeof(float));
                        else
                            sB[kk * BN + cc] = 0.f;
                    }
                } else {
                    float* sA = smA + ns * TK * BM;
                    for (int i = tid; i < TK * BM; i += 256) {
                        int r = i / BM, cc = i % BM;
                        int gk = nk * TK + r, gi = bm0 + cc;
                        if (gk < K && gi < N)
                            __pipeline_memcpy_async(sA + i, A + (size_t)gk * K + gi, sizeof(float));
                        else
                            sA[i] = 0.f;
                    }
                    float* sB = smB + ns * TK * BN;
                    for (int i = tid; i < TK * BN; i += 256) {
                        int r = i / BN, cc = i % BN;
                        int gk = nk * TK + r, gj = bn0 + cc;
                        if (gk < K && gj < N)
                            __pipeline_memcpy_async(sB + i, B + (size_t)gk * N + gj, sizeof(float));
                        else
                            sB[i] = 0.f;
                    }
                }
            }
            __pipeline_commit();
        }

        __pipeline_wait_prior(0);
        __syncthreads();

        // ---- 写回 + 清累加器（C 全矩阵覆盖；死元素 acc=0 与探针清零语义一致）----
#pragma unroll
        for (int p = 0; p < TP; p++) {
            const int gidx = p * 256 + tid;
            const int trow = (gidx / (BN / TN)) * TM;
            const int tcol = (gidx % (BN / TN)) * TN;
#pragma unroll
            for (int i = 0; i < TM; i++) {
                const int gr = bm0 + trow + i;
                if (gr >= N) continue;
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

// ---------------- naive 参考（探针语义：C 全矩阵 + k 三角守卫） ----------------
__global__ void syrk_naive(const float* __restrict__ A, float* __restrict__ C, int N, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * N;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < total; i += stride) {
        int r = (int)(i / N), c = (int)(i % N);
        float s = 0.f;
        const float* ar = A + (size_t)r * K;
        const float* ac = A + (size_t)c * K;
        for (int k = 0; k < c; k++) s += ar[k] * ac[k];   // k < j（列守卫）
        C[i] = s;
    }
}

__global__ void symm_naive(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int N, int K) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * N;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < total; i += stride) {
        int r = (int)(i / N), c = (int)(i % N);
        float s = 0.f;
        for (int k = 0; k < r; k++) s += A[(size_t)k * K + r] * B[(size_t)k * N + c];  // k < i（行守卫）
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
template <int BM, int BN, int TK, int S, int MODE>
double bench(const float* dA, const float* dB, float* dC, int N, int K) {
    const int smem = (S + 1) * (BM * TK + TK * BN) * 4;
    static bool attr_done = false;
    if (!attr_done) {
        cudaError_t e = cudaFuncSetAttribute(stril_tiled<BM, BN, TK, S, MODE>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_CAP);
        if (e != cudaSuccess) {
            printf("  [warn] cudaFuncSetAttribute failed (%dx%d/%d/%d/m%d)\n", BM, BN, TK, S, MODE);
            return -1.0;
        }
        attr_done = true;
    }
    dim3 grid(NUM_SMS), block(256);
    for (int w = 0; w < 3; w++)
        stril_tiled<BM, BN, TK, S, MODE><<<grid, block, smem>>>(dA, dB, dC, N, K);
    cudaDeviceSynchronize();
    // 【保险】launch 失败必须拦截（11-14 heat1d_oracle 假数据教训）
    {
        cudaError_t e2 = cudaGetLastError();
        if (e2 != cudaSuccess) {
            printf("  [fail] launch error (%dx%d/%d/%d/m%d): %s\n", BM, BN, TK, S, MODE,
                   cudaGetErrorString(e2));
            return -1.0;
        }
    }

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int it = 0; it < 20; it++)
        stril_tiled<BM, BN, TK, S, MODE><<<grid, block, smem>>>(dA, dB, dC, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (double(ms) / 20.0) * 1000.0;  // us
}

// ---------------- dispatch 宏（MODE 进模板实参） ----------------
#define DISPATCH_S(BM, BN, TK)                                                          \
    if (S == 1) us = bench<BM, BN, TK, 1, MODE>(dA, dB, dC, N, K);                      \
    else if (S == 2) us = bench<BM, BN, TK, 2, MODE>(dA, dB, dC, N, K);                 \
    else if (S == 3) us = bench<BM, BN, TK, 3, MODE>(dA, dB, dC, N, K);                 \
    else if (S == 4) us = bench<BM, BN, TK, 4, MODE>(dA, dB, dC, N, K);

#define DISPATCH_TK(BM, BN)                       \
    if (TK == 16) { DISPATCH_S(BM, BN, 16) }      \
    else if (TK == 32) { DISPATCH_S(BM, BN, 32) } \
    else if (TK == 64) { DISPATCH_S(BM, BN, 64) }

#define DISPATCH_BN(BM)                            \
    if (BN == 64) { DISPATCH_TK(BM, 64) }          \
    else if (BN == 128) { DISPATCH_TK(BM, 128) }   \
    else if (BN == 256) { DISPATCH_TK(BM, 256) }

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

// ---------------- 扫参主体（MODE 编译期分实例，只实例化本模式 48 配置） ----------------
template <int MODE>
void run_sweep(const float* dA, const float* dB, float* dC, int N, int K,
               int mBM, int mBN, int mTK, int mS) {
    const double gflop = (double)N * N * N / 1e12;   // 有用浮点 = 2·N³/2 = N³（守卫砍半）

    Row rows[48];
    int nRows = 0;
    double bestUs = 1e18; int bBM = 0, bBN = 0, bTK = 0, bS = 0;

    const int BMs[3] = {64, 128, 256};
    const int BNs[3] = {64, 128, 256};
    const int TKs[3] = {16, 32, 64};
    const int Ss[4] = {1, 2, 3, 4};

    for (int bi = 0; bi < 3; bi++) {
        for (int bj = 0; bj < 3; bj++) {
            int BM = BMs[bi], BN = BNs[bj];
            if (BM == 256 && BN == 256) continue;  // 寄存器约束排除（与模型同口径）
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

    printf("\n=== 最优: BM=%d BN=%d TK=%d S=%d @ %.1f us (%.1f TFLOPS, 有用浮点口径) ===\n",
           bBM, bBN, bTK, bS, bestUs, gflop / (bestUs * 1e-6));

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
}

int main(int argc, char** argv) {
    // 用法: ./stril_tiled <mode> [SIZE] [BM BN TK S]
    //   mode: 0=syrk 1=symm（必填第一个参数）
    //   nv==1: 仅 mode（SZ=512 + 模型标注缺省）
    //   nv==2: mode + SIZE
    //   nv==6: mode + SIZE + 模型决策四元组
    int mode = -1, SZ = 512;
    int mBM = 0, mBN = 0, mTK = 0, mS = 0;
    int vals[6] = {0, 0, 0, 0, 0, 0};
    int nv = 0;
    for (int i = 1; i < argc && nv < 6; ++i) vals[nv++] = atoi(argv[i]);
    if (nv >= 1) mode = vals[0];
    if (nv == 2) SZ = vals[1];
    else if (nv == 6) {
        SZ = vals[1];
        mBM = vals[2]; mBN = vals[3]; mTK = vals[4]; mS = vals[5];
    } else if (nv != 1) {
        printf("[参数] 用法: ./stril_tiled <mode: 0=syrk 1=symm> [SIZE] [BM BN TK S]\n");
        return 1;
    }
    if (mode != 0 && mode != 1) {
        printf("[参数] 用法: ./stril_tiled <mode: 0=syrk 1=symm> [SIZE] [BM BN TK S]\n");
        return 1;
    }
    if (mBM == 0) {   // 模型标注缺省 = 各模式决策（v1.8.3 后模型决策，11-15 更新）
        if (mode == 0) { mBM = 64; mBN = 64; mTK = 32; mS = 4; }   // syrk：双面板修正后回归 gemm 同参
        else           { mBM = 64; mBN = 64; mTK = 32; mS = 4; }   // symm
    }
    printf("[参数] mode=%s, SZ=%d, 模型标注: %d,%d,%d,%d\n",
           mode == 0 ? "syrk" : "symm", SZ, mBM, mBN, mTK, mS);

    const int N = SZ, K = SZ;
    const size_t bytes = (size_t)N * K * 4;
    float *dA, *dB, *dC, *dCRef;
    cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes); cudaMalloc(&dCRef, bytes);

    float* hA = (float*)malloc(bytes);
    for (size_t i = 0; i < (size_t)N * K; i++) hA[i] = (float)(i % 7) * 0.1f;
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    float* hB = (float*)malloc(bytes);
    for (size_t i = 0; i < (size_t)N * K; i++) hB[i] = (float)(i % 5) * 0.2f;
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);
    free(hA); free(hB);

    printf("=== %s 三角守卫 oracle (N=K=%d, FP32, 128 blocks 波次语义, chunk 级三角跳过) ===\n",
           mode == 0 ? "syrk" : "symm", N);
    printf("  载具语义 = 探针形态（C 全矩阵 + k 守卫; syrk 列守卫 k<j / symm 行守卫 k<i）\n");
    printf("  候选空间: BM/BN∈{64,128,256}², TK∈{16,32,64}, S∈{1..4}, smem≤99KB\n");
    printf("  排除: 256x256（寄存器硬上限）; 有用浮点口径 = N³（守卫砍半）\n\n");

    // ---- 正确性校验（naive vs tiled, 64/64/16/1; 累加顺序全同 → 预期 rel≈0）----
    {
        int threads = 256, blocks = (int)((N * N) / 256);
        if (mode == 0) syrk_naive<<<blocks, threads>>>(dA, dCRef, N, K);
        else           symm_naive<<<blocks, threads>>>(dA, dB, dCRef, N, K);
        cudaDeviceSynchronize();
        if (mode == 0)
            stril_tiled<64, 64, 16, 1, 0><<<NUM_SMS, 256, (2) * (64 * 16 + 16 * 64) * 4>>>(dA, dB, dC, N, K);
        else
            stril_tiled<64, 64, 16, 1, 1><<<NUM_SMS, 256, (2) * (64 * 16 + 16 * 64) * 4>>>(dA, dB, dC, N, K);
        cudaDeviceSynchronize();
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("正确性校验: launch FAIL (%s)\n\n", cudaGetErrorString(e));
            return 1;
        }

        double *d_sum2, *d_ref2, h_sum2 = 0, h_ref2 = 0;
        cudaMalloc(&d_sum2, 8); cudaMalloc(&d_ref2, 8);
        cudaMemset(d_sum2, 0, 8); cudaMemset(d_ref2, 0, 8);
        err_kernel<<<1024, 256>>>(dC, dCRef, (long long)N * N, d_sum2, d_ref2);
        cudaMemcpy(&h_sum2, d_sum2, 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_ref2, d_ref2, 8, cudaMemcpyDeviceToHost);
        double rel = (h_ref2 > 0) ? __builtin_sqrt(h_sum2 / h_ref2) : 0;
        printf("正确性校验(守卫形态, 全矩阵): %s (rel_err = %.2e)\n\n", rel < 1e-3 ? "PASS" : "FAIL", rel);
        cudaFree(d_sum2); cudaFree(d_ref2);
    }

    if (mode == 0) run_sweep<0>(dA, dB, dC, N, K, mBM, mBN, mTK, mS);
    else           run_sweep<1>(dA, dB, dC, N, K, mBM, mBN, mTK, mS);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dCRef);
    return 0;
}
