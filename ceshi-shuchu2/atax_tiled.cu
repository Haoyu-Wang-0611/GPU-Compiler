// atax_tiled.cu —— pass-chain GEMV 穷举对表 oracle（11-10，第一批③）
// 与 atax.mlir 同语义：pass1 tmp = A·x（行 GEMV）；pass2 y = A^T·tmp（列 GEMV）。
//   pass2 为转置访存（列方向）——pass-chain 类的真实压力测试。
// 参数空间 = 模型候选空间：T ∈ {8,16,32,64,128,256} × TK ∈ {16,32,64} × S ∈ {1..4}
//
// 用法: ./atax_tiled [N] [modelT] [modelTK] [modelS]
//   无参     → N=8192 无标注
//   4 参数   → N + 模型决策标注 (modelT, modelTK, modelS)
// （N 需为 4 的倍数且被所有 T/TK 整除——8192 满足；16B 对齐依赖此）
//
// 预注册要点（11-10，AI 写死）：
//   - N=8192：A = 256 MB >> L2 → 真 DRAM 流区，launch 可忽略（每 rep 2 发射）
//   - 模型决策（pass-chain 公式仿真，与 L1Feas 逐项一致）：
//     N=512 → (T=8,TK=16,S=4) @ 1.112178e4（与 11-04 硬件输出逐位吻合=仿真器互证）
//     N=8192 → (T=64,TK=16,S=1) @ 5.968949e5 ——决策随尺寸自适应翻转
//     （T=64 时 copy=577 ns 支配、expo 被隐藏 → S 并列回落 1）
//   - 已登记模型缺口：pass2 列方向的 sector 浪费未建模（T=8 时 ~4× 流量放大，
//     T≥64 段长 ≥256B 满线无浪费）→ 偏差方向偏向大 T，有利于模型决策；如实记录
//   - 预测列已删除（D3，11-14：旧公式与 C++ 主程序抄岔，论文只引实测名次）；判据 = rank 前 20% + 距优 ≤20%
//
// 编译/运行：
//   nvcc -O3 -arch=sm_89 atax_tiled.cu -o atax_tiled
//   CUDA_VISIBLE_DEVICES=1 ./atax_tiled 8192 64 16 1

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define COEF 1.0f

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ---- 结构可行性检查（D3 修复，11-14）：仅保留 smem/半机守卫两项结构性约束，
//   原 model_predict 的旧版时间公式（V0.3 前的 max 形式）已删除——与 C++ 主程序
//   公式抄岔（DeepSeek #4），论文不得引用已删除的预测列；时间口径以 L1Feas
//   决策输出（v1.7.1+ 两态式）为准，oracle 只报实测名次。
static bool model_feasible(int N, int T, int TK, int S) {
    const double smemCap = 99 * 1024;
    const int numSms = 128;
    // L1Feas v1.8.5 per-launch 口径（11-15 对齐）：单 launch = A 面板 T×TK
    //   （列向 pass 为 TK×T，等元素数）+ 向量 chunk TK，×(S+1)
    //   （旧式含 2·T 附加项 = 标注口径与 kernel 实际分配漂移，D3 教训同类）
    double smem_model = ((double)T * TK + TK) * 4 * (S + 1);
    if (smem_model > smemCap) return false;
    long long numBlocks = ((long long)N + T - 1) / T;
    if (numBlocks < numSms / 2) return false;              // 半机守卫
    return true;
}

// ---- pass1：行 GEMV（out[row] = Σ_j A[row][j]×x[j]），cp.async (S+1) 槽 ----
__global__ void gemv_rows(const float *__restrict__ A, const float *__restrict__ x,
                          float *__restrict__ out, int T, int TK, int S, int nSize) {
    extern __shared__ float slots[];
    int slotFloats = T * TK + TK;                 // [A-chunk T×TK | x-chunk TK]
    int tid = threadIdx.x, nth = blockDim.x;
    int row0 = blockIdx.x * T;
    int chunks = nSize / TK;                      // N % TK == 0（8192 满足）

    auto prefetch = [&](int w) {
        int j0 = w * TK;
        float *dstA = slots + (w % (S + 1)) * slotFloats;
        float *dstX = dstA + T * TK;
        int ops = TK / 4;                          // 每行段 TK float = TK/4 次 16B
        for (int k = tid; k < T * ops; k += nth) {
            int r = k / ops, q = k % ops;
            __pipeline_memcpy_async(dstA + r * TK + q * 4,
                                    A + (size_t)(row0 + r) * nSize + j0 + q * 4, 16);
        }
        for (int k = tid; k < ops; k += nth)
            __pipeline_memcpy_async(dstX + k * 4, x + j0 + k * 4, 16);
        __pipeline_commit();
    };

    // 【11-10 修复】acc 跨 chunk 累加（线程 tid 持有行 row0+tid，T ≤ 256 = nth）。
    // 原版 acc 在 chunk 循环内声明 → 每 chunk 清零 + 覆盖写 → out = 末 chunk
    // 部分和（1/chunks）→ 全配置 FAIL。修复与 pass2 对称（acc 循环外驻留）。
    float acc = 0.f;
    int issued = 0;
    for (int k = 0; k <= S && k < chunks; ++k) { prefetch(k); ++issued; }
    for (int w = 0; w < chunks; ++w) {
        __pipeline_wait_prior((unsigned)(issued - w - 1));
        __syncthreads();
        const float *bA = slots + (w % (S + 1)) * slotFloats;
        const float *bX = bA + T * TK;
        if (tid < T) {
            const float *bAr = bA + tid * TK;
            for (int c = 0; c < TK; ++c) acc += bAr[c] * bX[c];
        }
        __syncthreads();
        int next = w + S + 1;
        if (next < chunks) { prefetch(next); ++issued; }
    }
    if (tid < T) out[row0 + tid] = acc;
}

// ---- pass2：列 GEMV（out[col] = Σ_i A[i][col]×v[i]），转置访存 ----
__global__ void gemv_cols(const float *__restrict__ A, const float *__restrict__ v,
                          float *__restrict__ out, int T, int TK, int S, int nSize) {
    extern __shared__ float slots[];
    int slotFloats = TK * T + TK;                 // [A-panel TK×T | v-chunk TK]
    int tid = threadIdx.x, nth = blockDim.x;
    int col0 = blockIdx.x * T;
    int chunks = nSize / TK;
    float acc = 0.f;                               // 线程 c 跨 chunk 寄存器累加

    auto prefetch = [&](int w) {
        int i0 = w * TK;
        float *dstA = slots + (w % (S + 1)) * slotFloats;
        float *dstV = dstA + TK * T;
        int ops = T / 4;                           // 每列段 T float = T/4 次 16B
        for (int k = tid; k < TK * ops; k += nth) {
            int r = k / ops, q = k % ops;
            __pipeline_memcpy_async(dstA + r * T + q * 4,
                                    A + (size_t)(i0 + r) * nSize + col0 + q * 4, 16);
        }
        for (int k = tid; k < TK / 4; k += nth)
            __pipeline_memcpy_async(dstV + k * 4, v + i0 + k * 4, 16);
        __pipeline_commit();
    };

    int issued = 0;
    for (int k = 0; k <= S && k < chunks; ++k) { prefetch(k); ++issued; }
    for (int w = 0; w < chunks; ++w) {
        __pipeline_wait_prior((unsigned)(issued - w - 1));
        __syncthreads();
        const float *bA = slots + (w % (S + 1)) * slotFloats;
        const float *bV = bA + TK * T;
        if (tid < T) {                             // 线程 c 负责一列，smem 合并访问
            for (int r = 0; r < TK; ++r) acc += bA[r * T + tid] * bV[r];
        }
        __syncthreads();
        int next = w + S + 1;
        if (next < chunks) { prefetch(next); ++issued; }
    }
    if (tid < T) out[col0 + tid] = acc;
}

int main(int argc, char** argv) {
    // 用法: ./atax_tiled [N] [modelT] [modelTK] [modelS]
    int NN = 8192, MT = -1, MTK = -1, MS = -1;
    int vals[4] = {0, 0, 0, 0}, nv = 0;
    for (int i = 1; i < argc && nv < 4; ++i) vals[nv++] = atoi(argv[i]);
    if (nv >= 1) NN = vals[0];
    if (nv >= 4) { MT = vals[1]; MTK = vals[2]; MS = vals[3]; }
    if (NN % 4 != 0) { printf("[参数] N 必须为 4 的倍数（16B 对齐），当前 %d\n", NN); return 1; }
    printf("[参数] N=%d, 模型标注: T=%d TK=%d S=%d\n", NN, MT, MTK, MS);

    size_t bA = (size_t)NN * NN * sizeof(float), bV = (size_t)NN * sizeof(float);
    float *hA = (float *)malloc(bA), *hx = (float *)malloc(bV);
    float *htmp = (float *)malloc(bV), *hy = (float *)malloc(bV);
    float *rtmp = (float *)malloc(bV), *ry = (float *)malloc(bV);
    for (int i = 0; i < NN * NN; ++i) hA[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < NN; ++i) hx[i] = (float)(rand() % 100) / 100.0f;
    // CPU double 参考（两 pass）
    for (int i = 0; i < NN; ++i) {
        double s = 0;
        for (int j = 0; j < NN; ++j) s += (double)hA[(size_t)i * NN + j] * hx[j];
        rtmp[i] = (float)s;
    }
    for (int j = 0; j < NN; ++j) {
        double s = 0;
        for (int i = 0; i < NN; ++i) s += (double)hA[(size_t)i * NN + j] * rtmp[i];
        ry[j] = (float)s;
    }

    float *dA, *dx, *dtmp, *dy;
    CHECK(cudaMalloc(&dA, bA)); CHECK(cudaMalloc(&dx, bV));
    CHECK(cudaMalloc(&dtmp, bV)); CHECK(cudaMalloc(&dy, bV));
    CHECK(cudaMemcpy(dA, hA, bA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, hx, bV, cudaMemcpyHostToDevice));
    CHECK(cudaFuncSetAttribute(gemv_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
    CHECK(cudaFuncSetAttribute(gemv_cols, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));

    struct Row { int T, TK, S; double meas; };
    std::vector<Row> rows;
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));

    printf("=== atax 穷举对表 (N=%d, A=%.0fMB, 2 pass GEMV: 行+列转置) ===\n",
           NN, bA / 1048576.0);

    for (int T = 8; T <= 256; T *= 2)
    for (int TK = 16; TK <= 64; TK *= 2)
    for (int S = 1; S <= 4; ++S) {
        bool modelOK = model_feasible(NN, T, TK, S);
        if (!modelOK) { printf("T=%-4d TK=%-3d S=%d  模型拒（smem/半机守卫）——跳过\n", T, TK, S); continue; }
        int tilesN = NN / T;                        // N % T == 0（8192 满足）
        int smem = (S + 1) * (T * TK + TK) * 4;
        if (smem > 101376) { printf("T=%-4d TK=%-3d S=%d  kernel smem=%5.1fKB 超限跳过\n",
                                    T, TK, S, smem / 1024.0); continue; }
        int grid = tilesN;

        // 正确性：pass1 → pass2，对 CPU double 参考（抽 500 点/pass）
        CHECK(cudaMemset(dtmp, 0, bV)); CHECK(cudaMemset(dy, 0, bV));
        gemv_rows<<<grid, 256, smem>>>(dA, dx, dtmp, T, TK, S, NN);
        CHECK(cudaDeviceSynchronize());
        gemv_cols<<<grid, 256, smem>>>(dA, dtmp, dy, T, TK, S, NN);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(htmp, dtmp, bV, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hy, dy, bV, cudaMemcpyDeviceToHost));
        double maxRel = 0;
        for (int i = 0; i < NN; i += std::max(1, NN / 500)) {
            double r1 = fabs((double)htmp[i] - (double)rtmp[i]) / (fabs((double)rtmp[i]) + 1e-6);
            double r2 = fabs((double)hy[i] - (double)ry[i]) / (fabs((double)ry[i]) + 1e-6);
            if (r1 > maxRel) maxRel = r1;
            if (r2 > maxRel) maxRel = r2;
        }
        bool ok = maxRel < 5e-3;

        // 计时：3 rep × (pass1+pass2)，事件对累计
        double bestRep = 1e18;
        for (int rep = 0; rep < 3; ++rep) {
            float ms = 0;
            CHECK(cudaEventRecord(ev0));
            gemv_rows<<<grid, 256, smem>>>(dA, dx, dtmp, T, TK, S, NN);
            gemv_cols<<<grid, 256, smem>>>(dA, dtmp, dy, T, TK, S, NN);
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaEventSynchronize(ev1));
            CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            if (ms < bestRep) bestRep = ms;
        }
        double us = bestRep * 1000.0;
        printf("T=%-4d TK=%-3d S=%d  smem=%5.1fKB  实测: %12.1f us  %s\n",
               T, TK, S, smem / 1024.0, us, ok ? "PASS" : "FAIL");
        rows.push_back({T, TK, S, us});
    }

    // ---------------- 排名 + 判定 ----------------
    printf("\n===== 实测排名（升序）=====\n");
    std::vector<Row> sorted = rows;
    for (size_t i = 0; i < sorted.size(); ++i)
        for (size_t j = i + 1; j < sorted.size(); ++j)
            if (sorted[j].meas < sorted[i].meas) std::swap(sorted[i], sorted[j]);
    int modelRank = -1; double bestMeas = sorted[0].meas;
    for (size_t i = 0; i < sorted.size(); ++i) {
        bool isModel = (sorted[i].T == MT && sorted[i].TK == MTK && sorted[i].S == MS);
        if (isModel) modelRank = (int)i + 1;
        printf("%2zd   T=%-4d TK=%-3d S=%d   %12.1f us%s\n", i + 1, sorted[i].T,
               sorted[i].TK, sorted[i].S, sorted[i].meas, isModel ? "  <<<< 模型决策" : "");
    }
    if (modelRank > 0) {
        double modelMeas = 0;
        for (auto &r : sorted)
            if (r.T == MT && r.TK == MTK && r.S == MS) modelMeas = r.meas;
        double pct = 100.0 * modelRank / (double)sorted.size();
        double gap = 100.0 * (modelMeas - bestMeas) / bestMeas;
        printf("\n[判定] 模型决策 (T=%d,TK=%d,S=%d): 穷举第 %d/%d 名 = 前 %.1f%%，距最优 %+.1f%% → %s\n",
               MT, MTK, MS, modelRank, (int)sorted.size(), pct, gap,
               (pct <= 20.0 && gap <= 20.0) ? "前20% 达标" : "前20% 未达");
    } else {
        printf("\n[判定] 模型决策未在可行集中出现（标注参数或缺测）\n");
    }

    free(hA); free(hx); free(htmp); free(hy); free(rtmp); free(ry);
    CHECK(cudaFree(dA)); CHECK(cudaFree(dx)); CHECK(cudaFree(dtmp)); CHECK(cudaFree(dy));
    return 0;
}
