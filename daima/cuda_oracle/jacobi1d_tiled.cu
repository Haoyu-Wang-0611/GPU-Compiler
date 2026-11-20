// jacobi1d_tiled.cu —— 1D stencil 穷举对表 oracle（v2，11-08：尺寸参数化）
// 与 jacobi1d.mlir 同语义：B[i] = (A[i-1]+A[i]+A[i+1]) × 1/3，interior = [1, N-2]，
//   兄弟交换结构（每时间步 B=f(A) 后 A=f(B)）→ host 侧 2×STEPS 次发射 + 指针交换
// 参数空间：T ∈ {16,32,...,≤N} × S ∈ {1..4}（smem 超限自动跳过）
//
// 用法: ./jacobi1d_tiled [N] [modelT] [modelS]
//   无参            → N=1000, 模型决策 16/1（L1Feas 11-02 输出）
//   ./jacobi1d_tiled 65536              → 大尺寸全扫参
//   ./jacobi1d_tiled 65536 32 4         → 大尺寸 + 标注模型决策
// （N 需为偶数——窗口钳位按偶数对齐）
//
// 教训入册（11-08）：N=1000 时 numTiles ≤ 63 < 128 恒单波 → launch 开销主导
//   （20 发射 × ~7µs），排名为噪声；legacy 模型未算逐发射 launch 开销（33× 差）。
//   大尺寸（65536）下 numTiles=4096 → 32 waves → S 维恢复物理意义。
//
// 编译/运行：
//   nvcc -O3 -arch=sm_89 jacobi1d_tiled.cu -o jacobi1d_tiled
//   CUDA_VISIBLE_DEVICES=1 ./jacobi1d_tiled 65536

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#ifndef STEPS
#define STEPS 9   // 与 jacobi1d.mlir 一致（t = 1 to 10 → trip 9；11-09 复查修正）
#endif
#define COEF 0.33333334f    // MLIR: 3.333333e-01

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ---------------- legacy 1D 模型公式（决策出处；base 项随尺寸） ----------------
static double model_predict(int NN, int T, int S) {
    const double smBw = 7.21, rExposed = 330.0, aT = 0.60, pExp = 1.20,
                 deltaFloor = 52.0, launch = 3000.0;
    const int numSms = 128; const double smemCap = 99 * 1024;
    // legacy bytes = Σ(base + (T-1)*elem)。【11-08 修正】legacy 的 base 项
    // （全数组大小）在大 N 不可复现且会把全部配置判超限 → 置 0，
    // 预测列只保留 (T-1)*4 扩展项；可行性由 kernel 实际 smem 把关。
    double bytes = 2.0 * ((double)(T - 1) * 4);
    double smem = bytes * (S + 1);
    if (smem > smemCap) return -1.0;
    long long numTiles = ((long long)(NN - 2) + T - 1) / T;
    double waves = (double)((numTiles + numSms - 1) / numSms);
    double eff  = std::min((double)S, std::max(1.0, waves - 1.0));
    double copy = bytes / smBw;
    double R    = std::max(rExposed - aT * T, 1.0);
    double expo = std::max(R / std::pow(eff, pExp), deltaFloor);
    return launch + waves * (copy + expo);
}

// ---------------- kernel：单 pass（in → out），cp.async (S+1) 窗口槽 ----------------
__global__ void jacobi1d_pass(const float *__restrict__ in, float *__restrict__ out,
                              int Ti, int S, int tilesN, int tilesPerBlock, int nSize) {
    extern __shared__ float slots[];
    int slotFloats = Ti + 2;
    int tid = threadIdx.x, nth = blockDim.x;
    long long numTiles = (long long)tilesN;
    int tileLo = (int)((long long)blockIdx.x * tilesPerBlock);
    int tileHi = min((long long)(tileLo + tilesPerBlock), numTiles) - 1;
    if (tileHi < tileLo) return;
    int count = tileHi - tileLo + 1;

    auto prefetch = [&](int w) {
        int start = (tileLo + w) * Ti;               // 窗口左端 = tile 左界 −1（Ti 偶 → start 偶）
        float *dst0 = slots + (w % (S + 1)) * slotFloats;
        int ops_total = (Ti + 2) / 2;                // 1D 连续窗口：(Ti+2) float = ops_total 次 8B
        for (int k = tid; k < ops_total; k += nth) {
            int g0 = start + k * 2;                  // 本 8B 对首元素全局下标（偶）
            // 整对越界才钳到 nSize-2（偶，保对齐）；合法对不钳（gi 的真邻居不可钳）
            int idx = (g0 >= nSize) ? (nSize - 2) : g0;
            float *dst = dst0 + k * 2;
            __pipeline_memcpy_async(dst, in + idx, 8);
        }
        __pipeline_commit();
    };

    int issued = 0;
    for (int k = 0; k <= S && k < count; ++k) { prefetch(k); ++issued; }
    for (int w = 0; w < count; ++w) {
        __pipeline_wait_prior((unsigned)(issued - w - 1));
        __syncthreads();
        const float *base = slots + (w % (S + 1)) * slotFloats;
        int gLo = 1 + (tileLo + w) * Ti;
        for (int e = tid; e < Ti; e += nth) {
            int gi = gLo + e;
            if (gi > nSize - 2) continue;            // interior 上界 = N-2
            const float *p = base + e + 1;           // 窗口内中心位置
            out[gi] = (p[-1] + p[0] + p[1]) * COEF;  // 左/中/右，与 MLIR 同序
        }
        __syncthreads();
        int next = w + S + 1;
        if (next < count) { prefetch(next); ++issued; }
    }
}

int main(int argc, char** argv) {
    // 用法: ./jacobi1d_tiled [N] [modelT] [modelS]
    //   0 参数  → N=1000，模型决策 16/1
    //   1 参数  → 指定 N 全扫参
    //   3 参数  → 指定 N + 标注模型决策 (modelT, modelS)
    int NN = 1000, MODEL_T = 16, MODEL_S = 1;
    int vals[3] = {0, 0, 0}, nv = 0;
    for (int i = 1; i < argc && nv < 3; ++i) vals[nv++] = atoi(argv[i]);
    if (nv >= 1) NN = vals[0];
    if (nv >= 2) MODEL_T = vals[1];
    if (nv >= 3) MODEL_S = vals[2];
    if (NN % 2 != 0) { printf("[参数] N 必须为偶数（窗口钳位对齐），当前 %d\n", NN); return 1; }
    const int INT_LO = 1, INT_HI = NN - 2;
    printf("[参数] N=%d, interior=[%d,%d], 模型决策: Tile=%d Stages=%d\n",
           NN, INT_LO, INT_HI, MODEL_T, MODEL_S);

    size_t bytes = (size_t)NN * sizeof(float);
    float *hA = (float *)malloc(bytes), *hB = (float *)malloc(bytes), *hRef = (float *)malloc(bytes);
    for (int i = 0; i < NN; ++i) hA[i] = (float)(rand() % 100) / 100.0f;
    memset(hB, 0, bytes);

    // CPU 单 pass 参考（interior，double）
    for (int i = INT_LO; i <= INT_HI; ++i) {
        double s = (double)hA[i - 1] + hA[i] + hA[i + 1];
        hRef[i] = (float)(s * (double)COEF);
    }

    float *dA, *dB;
    CHECK(cudaMalloc(&dA, bytes)); CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dB, 0, bytes));
    CHECK(cudaFuncSetAttribute(jacobi1d_pass, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));

    struct Row { int T, S; double meas, pred; };
    std::vector<Row> rows;
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));

    printf("=== jacobi-1d 穷举对表 (N=%d, interior %d, %d 时间步 x 2 pass, 兄弟交换语义) ===\n",
           NN, INT_HI, STEPS);

    for (int T = 16; T <= NN; T *= 2)
    for (int S = 1; S <= 4; ++S) {
        double pred = model_predict(NN, T, S);
        if (pred < 0) { printf("T=%-6d S=%d  smem 超限跳过（模型同拒）\n", T, S); continue; }
        int tilesN = (INT_HI - INT_LO + 1 + T - 1) / T;
        long long numTiles = (long long)tilesN;
        int grid = (int)std::min((long long)128, numTiles);
        int per = (int)((numTiles + grid - 1) / grid);
        int smem = (S + 1) * (T + 2) * 4;
        if (smem > 101376) { printf("T=%-6d S=%d  kernel smem=%5.1fKB 超限跳过\n", T, S, smem / 1024.0); continue; }

        // 正确性：单 pass vs CPU（先恢复 dA——计时交换链会写脏设备侧输入）
        CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dB, 0, bytes));
        jacobi1d_pass<<<grid, 256, smem>>>(dA, dB, T, S, tilesN, per, NN);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hB, dB, bytes, cudaMemcpyDeviceToHost));
        double maxRel = 0; int cnt = 0;
        for (int i = INT_LO; i <= INT_HI; i += std::max(1, (INT_HI - INT_LO + 1) / 500)) {
            double rel = fabs((double)hB[i] - (double)hRef[i]) / (fabs((double)hRef[i]) + 1e-6);
            if (rel > maxRel) maxRel = rel;
            ++cnt;
        }
        bool ok = maxRel < 1e-4;

        // 计时：3 rep × (STEPS×2) 次发射（兄弟交换），逐 launch 事件累计
        double bestRep = 1e18;
        for (int rep = 0; rep < 3; ++rep) {
            float *cur = dA, *nxt = dB;
            float acc = 0;
            for (int s = 0; s < STEPS * 2; ++s) {
                CHECK(cudaEventRecord(ev0));
                jacobi1d_pass<<<grid, 256, smem>>>(cur, nxt, T, S, tilesN, per, NN);
                CHECK(cudaEventRecord(ev1));
                CHECK(cudaEventSynchronize(ev1));
                float ms = 0; CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
                acc += ms;
                std::swap(cur, nxt);
            }
            if (acc < bestRep) bestRep = acc;
        }
        double us = bestRep * 1000.0;
        printf("T=%-6d S=%d  smem=%5.1fKB  实测: %12.1f us  预测: %12.1f us  %s\n",
               T, S, smem / 1024.0, us, pred / 1000.0, ok ? "PASS" : "FAIL");
        rows.push_back({T, S, us, pred});
    }

    // ---------------- 排名 + 判定 ----------------
    printf("\n===== 实测排名（升序）=====\n");
    std::vector<Row> sorted = rows;
    for (size_t i = 0; i < sorted.size(); ++i)
        for (size_t j = i + 1; j < sorted.size(); ++j)
            if (sorted[j].meas < sorted[i].meas) std::swap(sorted[i], sorted[j]);
    int modelRank = -1; double bestMeas = sorted[0].meas;
    for (size_t i = 0; i < sorted.size(); ++i) {
        bool isModel = (sorted[i].T == MODEL_T && sorted[i].S == MODEL_S);
        if (isModel) modelRank = (int)i + 1;
        printf("%2zd   T=%-6d S=%d   %12.1f us%s\n", i + 1, sorted[i].T, sorted[i].S,
               sorted[i].meas, isModel ? "  <<<< 模型决策" : "");
    }
    if (modelRank > 0) {
        double modelMeas = 0;
        for (auto &r : sorted)
            if (r.T == MODEL_T && r.S == MODEL_S) modelMeas = r.meas;
        double pct = 100.0 * modelRank / (double)sorted.size();
        double gap = 100.0 * (modelMeas - bestMeas) / bestMeas;
        printf("\n[判定] 模型决策 (T=%d,S=%d): 穷举第 %d/%d 名 = 前 %.1f%%，距最优 %+.1f%% → %s\n",
               MODEL_T, MODEL_S, modelRank, (int)sorted.size(), pct, gap,
               (pct <= 20.0 && gap <= 20.0) ? "前20% 达标" : "前20% 未达");
    }

    free(hA); free(hB); free(hRef);
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB));
    return 0;
}
