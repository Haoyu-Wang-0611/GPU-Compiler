// jacobi2d_tiled.cu —— 2D stencil 穷举对表 oracle（11-08）
// 与 jacobi2d.mlir 同语义：B[i][j] = 0.2*(A[i][j]+A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])
//   （无交换形式：恒读 A 写 B，16 步等价重复——与发射目标 final.mlir 语义一致）
// 参数空间 = 模型候选空间：T ∈ {16,32,64,128} × S ∈ {1..4}（16 配置）
// smem 口径 = 修正后模型口径（v1.2.2）：只 A 窗口 ×(S+1)；行 stride pad 2 float（cp.async 8B 对齐）
// kernel 形态：grid = min(128, numTiles) 块，块内 grid-stride 多 tile，
//   cp.async 预取下一个 tile 的 A 窗口（S+1 个窗口槽）——模型 smem 公式的物理对应
//
// 编译/运行：
//   nvcc -O3 -arch=sm_89 jacobi2d_tiled.cu -o jacobi2d_tiled
//   CUDA_VISIBLE_DEVICES=1 ./jacobi2d_tiled
//
// 程序自动输出：
//   1. 16 配置扫参表：预测（内嵌模型公式，v1.2.2 口径）vs 实测
//   2. 模型决策（Tile2d=64,64/S=1 @ 预注册 1.6342e5 ns）的穷举排名 + 前 20% 判定
//   3. eff 封顶裁决：T=64 组 S=2/3/4 vs S=1 的实测比
//      （S 间无差 = waves 封顶假设获数据支持；S=4 显著快 = 封顶去除、V0.3 修正项）

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#ifndef N
#define N 1024              // 空间尺寸（MLIR: 1..N-1，interior = 1..N-2）
#endif
#define INT_LO 1
#define INT_HI (N - 2)      // 闭区间
#ifndef TSTEPS
#define TSTEPS 16
#endif
#define IDX(r, c) ((size_t)(r) * N + (c))

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ---------------- 模型公式（v1.2.2 口径，与 L1Feas.cpp stencil-2d 模板逐项对应） ----------------
struct Pred { bool feas; double ns; };
static Pred model_predict(int T, int S) {
    const double smBw = 7.21, smCompute = 630.0, rExposed = 330.0, aT = 0.60,
                 pExp = 1.20, deltaFloor = 52.0, launch = 3000.0;
    const int numSms = 128; const double smemCap = 99 * 1024;
    Pred p{false, 0.0};
    double fpA = (double)(T + 2) * (T + 2) * 4;      // A 读窗口（halo 并入）
    double fpB = (double)T * T * 4;                  // B 写 tile（无 halo）
    double copy = (fpA + fpB) / smBw;
    double comp = 5.0 * T * T / smCompute;           // FLOPs/iter = 5
    double smem  = fpA * (S + 1);                    // v1.2.2: 只读窗口进 smem
    if (smem > smemCap) return p;
    long long nb = (long long)((INT_HI - INT_LO + 1 + T - 1) / T) *
                   (long long)((INT_HI - INT_LO + 1 + T - 1) / T);
    if (nb < numSms / 2) return p;                   // 半机填充守卫
    double waves = (double)((nb + numSms - 1) / numSms);
    double eff  = std::min((double)S, std::max(1.0, waves - 1.0));
    double R    = std::max(rExposed - aT * T, 1.0);
    double expo = std::max(R / std::pow(eff, pExp), deltaFloor);
    p.ns = launch + (double)TSTEPS * waves * (copy + comp + expo);
    p.feas = true;
    return p;
}

// ---------------- kernel：grid-stride tile + cp.async (S+1) 窗口槽 ----------------
// 动态 smem 布局：(S+1) 个槽，槽 = (T+2) 行 × (T+4) float（行尾 pad 2 float → 8B 对齐）
__global__ void jacobi2d_tiled(const float *__restrict__ A, float *__restrict__ B,
                               int Ti, int S, int tilesX, int tilesPerBlock) {
    extern __shared__ float slots[];
    int slotFloats = (Ti + 2) * (Ti + 4);
    int rows = Ti + 2, ops = (Ti + 2) / 2;           // 每行 (Ti+2) float = ops 次 8B cp.async
    int tid = threadIdx.x, nth = blockDim.x;
    long long numTiles = (long long)tilesX * tilesX;
    int tileLo = (int)((long long)blockIdx.x * tilesPerBlock);
    int tileHi = min((long long)(tileLo + tilesPerBlock), numTiles) - 1;
    if (tileHi < tileLo) return;
    int count = tileHi - tileLo + 1;

    auto prefetch = [&](int w) {                      // tile w → slot w%(S+1)
        int bi = (tileLo + w) / tilesX, bj = (tileLo + w) % tilesX;
        int r0 = bi * Ti, c0 = bj * Ti;               // 窗口左上（= interior 左上 -1，偶数对齐）
        float *dst0 = slots + (w % (S + 1)) * slotFloats;
        for (int k = tid; k < rows * ops; k += nth) {
            int r = k / ops, o = k % ops;
            int rr = min(r0 + r, N - 1);              // OOB 尾部钳位（值不被有效输出使用）
            int cc = min(c0 + o * 2, N - 2);          // 钳到偶数（8B 对齐），同上
            float *dst = dst0 + r * (Ti + 4) + o * 2;
            const float *src = A + (size_t)rr * N + cc;
            __pipeline_memcpy_async(dst, src, 8);
        }
        __pipeline_commit();
    };

    int issued = 0;
    for (int k = 0; k <= S && k < count; ++k) { prefetch(k); ++issued; }   // prologue: tile 0..S
    for (int w = 0; w < count; ++w) {
        // 等待 tile w 的 commit 组完成（其后发射的组保持 pending）
        __pipeline_wait_prior((unsigned)(issued - w - 1));
        __syncthreads();
        const float *base = slots + (w % (S + 1)) * slotFloats;
        int bi = (tileLo + w) / tilesX, bj = (tileLo + w) % tilesX;
        for (int e = tid; e < Ti * Ti; e += nth) {
            int r = e / Ti, c = e % Ti;
            int gi = INT_LO + bi * Ti + r, gj = INT_LO + bj * Ti + c;
            if (gi > INT_HI || gj > INT_HI) continue;  // 尾部 tile 的出界元素跳过
            const float *p = base + (r + 1) * (Ti + 4) + (c + 1);
            float s = p[0] + p[-(Ti + 4)] + p[Ti + 4] + p[-1] + p[1];  // 中/上/下/左/右
            B[IDX(gi, gj)] = s * 0.2f;
        }
        __syncthreads();
        // 计算释放槽 w 后，才发射 tile w+S+1（其槽号 = (w+S+1)%(S+1) = w%(S+1)——
        // 恰是刚用完的槽；提前发射会与本次计算读写同槽 → 竞态，11-08 复查修复）
        int next = w + S + 1;
        if (next < count) { prefetch(next); ++issued; }
    }
}

int main() {
    srand(42);
    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float *)malloc(bytes), *hB = (float *)malloc(bytes), *hRef = (float *)malloc(bytes);
    for (size_t p = 0; p < (size_t)N * N; ++p) hA[p] = (float)(rand() % 100) / 100.0f;
    memset(hB, 0, bytes);

    // CPU 单步参考（interior，double 累加）
    for (int i = INT_LO; i <= INT_HI; ++i)
        for (int j = INT_LO; j <= INT_HI; ++j) {
            double s = (double)hA[IDX(i, j)] + hA[IDX(i - 1, j)] + hA[IDX(i + 1, j)]
                     + hA[IDX(i, j - 1)] + hA[IDX(i, j + 1)];
            hRef[IDX(i, j)] = (float)(s * 0.2);
        }

    float *dA, *dB;
    CHECK(cudaMalloc(&dA, bytes)); CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dB, 0, bytes));
    CHECK(cudaFuncSetAttribute(jacobi2d_tiled, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));

    const int cT[4] = {16, 32, 64, 128};
    const int MODEL_T = 64, MODEL_S = 1;   // L1Feas v1.2.2 决策（预注册 1.6342e5 ns）
    struct Row { int T, S; double meas, pred; bool feas; };
    Row rows[16]; int nr = 0;
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));

    printf("=== jacobi-2d 穷举对表 (N=%d, interior %d, %d 时间步, 无交换语义) ===\n", N, INT_HI, TSTEPS);
    printf("[预注册] 模型决策: Tile2d=64,64 Stages=1, predicted = 1.6342e5 ns\n");
    printf("[预注册] 可行性面(v1.2.2 口径): T=64 全 S 可行; T=128 仅 S=1; smem=fpA×(S+1)\n\n");

    for (int ti = 0; ti < 4; ++ti)
    for (int S = 1; S <= 4; ++S) {
        int T = cT[ti];
        Pred pr = model_predict(T, S);
        int smem = (S + 1) * (T + 2) * (T + 4) * 4;   // kernel 实际（行 pad，比模型多 ~1.5%）
        if (!pr.feas || smem > 101376) {
            printf("T=%-4d S=%d  smem=%6.1fKB  超限跳过（模型同拒）\n", T, S, smem / 1024.0);
            continue;
        }
        int tilesX = (INT_HI - INT_LO + 1 + T - 1) / T;
        long long numTiles = (long long)tilesX * tilesX;
        int grid = (int)std::min((long long)128, numTiles);
        int per = (int)((numTiles + grid - 1) / grid);

        // 正确性：单步 vs CPU
        CHECK(cudaMemset(dB, 0, bytes));
        jacobi2d_tiled<<<grid, 256, smem>>>(dA, dB, T, S, tilesX, per);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hB, dB, bytes, cudaMemcpyDeviceToHost));
        double maxRel = 0; int cnt = 0;
        for (int i = INT_LO; i <= INT_HI; i += 37)
            for (int j = INT_LO; j <= INT_HI; j += 37) {
                double rel = fabs((double)hB[IDX(i, j)] - (double)hRef[IDX(i, j)])
                           / (fabs((double)hRef[IDX(i, j)]) + 1e-6);
                if (rel > maxRel) maxRel = rel;
                ++cnt;
            }
        bool ok = maxRel < 1e-4;

        // 计时：3 rep × 16 步（逐 launch 事件累计 kernel 纯时间）
        double bestRep = 1e18;
        for (int rep = 0; rep < 3; ++rep) {
            float acc = 0;
            for (int s = 0; s < TSTEPS; ++s) {
                CHECK(cudaEventRecord(ev0));
                jacobi2d_tiled<<<grid, 256, smem>>>(dA, dB, T, S, tilesX, per);
                CHECK(cudaEventRecord(ev1));
                CHECK(cudaEventSynchronize(ev1));
                float ms = 0; CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
                acc += ms;
            }
            if (acc < bestRep) bestRep = acc;
        }
        double us = bestRep * 1000.0;
        printf("T=%-4d S=%d  smem=%6.1fKB  实测: %9.1f us  预测: %9.1f us  比值 %.2f  %s\n",
               T, S, smem / 1024.0, us, pr.ns / 1000.0, us / (pr.ns / 1000.0),
               ok ? "PASS" : "FAIL");
        rows[nr++] = {T, S, us, pr.ns / 1000.0, ok};
    }

    // ---------------- 排名 + 判定 ----------------
    printf("\n===== 实测排名（可行配置，升序）=====\n");
    Row sorted[16];
    memcpy(sorted, rows, sizeof(Row) * nr);
    for (int i = 0; i < nr; ++i)
        for (int j = i + 1; j < nr; ++j)
            if (sorted[j].meas < sorted[i].meas) std::swap(sorted[i], sorted[j]);
    int modelRank = -1; double bestMeas = sorted[0].meas;
    for (int i = 0; i < nr; ++i) {
        bool isModel = (sorted[i].T == MODEL_T && sorted[i].S == MODEL_S);
        if (isModel) modelRank = i + 1;
        printf("%2d   T=%-4d S=%d   %10.1f us%s\n", i + 1, sorted[i].T, sorted[i].S,
               sorted[i].meas, isModel ? "  <<<< 模型决策" : "");
    }
    if (modelRank > 0) {
        double modelMeas = 0;
        for (int i = 0; i < nr; ++i)
            if (sorted[i].T == MODEL_T && sorted[i].S == MODEL_S) modelMeas = sorted[i].meas;
        double pct = 100.0 * modelRank / nr;
        double gap = 100.0 * (modelMeas - bestMeas) / bestMeas;
        printf("\n[判定] 模型决策 (T=%d,S=%d): 穷举第 %d/%d 名 = 前 %.1f%%，距最优 %+.1f%% → %s\n",
               MODEL_T, MODEL_S, modelRank, nr, pct, gap,
               (pct <= 20.0 && gap <= 20.0) ? "前20% 达标" : "前20% 未达");
    }

    // ---------------- eff 封顶裁决（T=64 组内 S 维实测比） ----------------
    printf("\n===== eff 封顶裁决（T=64 组）=====\n");
    double m[5] = {0};
    for (int i = 0; i < nr; ++i) if (rows[i].T == 64) m[rows[i].S] = rows[i].meas;
    if (m[1] > 0 && m[4] > 0) {
        double gain = 100.0 * (m[1] - m[4]) / m[1];
        printf("T=64: S=1 %.1f | S=2 %.1f | S=3 %.1f | S=4 %.1f us → S=4 相对 S=1 %+.1f%%\n",
               m[1], m[2], m[3], m[4], -gain);
        if (gain > 2.0)
            printf("[裁决] S 维实测有真实收益 → eff 的 waves 封顶假设被否证 → V0.3 修正项\n");
        else
            printf("[裁决] S 维实测无显著差（≤2%%）→ waves 封顶假设获数据支持\n");
    }

    free(hA); free(hB); free(hRef);
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB));
    return 0;
}
