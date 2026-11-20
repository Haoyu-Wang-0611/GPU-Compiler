// bicg_tiled.cu —— bicg 穷举对表 oracle（11-15，A3 主表第 13 行）
// 与 bicg_probe.mlir 同语义（官方 PolyBench bicg 形态）：
//   pass1: s[j] += r[i]·A[i][j]  —— 列向 GEMV（out=j 内层，归约沿 A 行维；
//            A 面板 = TK×T k-major，行连续 16B 读合并）
//   pass2: q[i] += A[i][j]·p[j]  —— 行向 GEMV（与 mvt_tiled 的 gemv_rows_acc 逐字节同构）
// s/q 零初值累加（官方 bicg 语义）。
//
// 【协议】双判据（mvt_tiled 首创，照搬）：
//   ① 模型可行子集排名；② 全空间排名（含模型拒的 T=16..256）+ 半机守卫检验。
//   model_predict 仅复现 L1Feas GEMV 可行性口径（smem+半机守卫）用于子集标注——
//   预测时间值不参与判定（D3 教训：oracle 不复制模型时间公式）。
//
// 【预注册（11-15，动码前写死）】模型决策 (T=8, TK=64, S=1)，predicted 2.616888e+04 ns
//   （与 L1Feas v1.8 决策及 mvt 行逐位同值）；判据：双排名均前 20% + 距优 ≤20%。
//   预期：流量结构与 mvt 同（A 流两次）→ 排名行为镜像 mvt（全空间 rank ~1 量级）。
//   风险登记：T=8 线程效率损失未建模（mvt 同款）；列向面板的 smem 读为跨行步进
//   （stride T），与行向不同——若排名劣于 mvt，先查此差异。
//
// 编译/运行：
//   nvcc -O3 -arch=sm_89 bicg_tiled.cu -o bicg_tiled
//   CUDA_VISIBLE_DEVICES=1 ./bicg_tiled 512 8 64 1
//
// 【保险继承】launch 失败拦截 + cudaFuncSetAttribute（11-14 heat1d 教训）。

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ---- L1Feas GEMV 可行性口径复现（仅用于子集标注；预测时间不参与判定） ----
static double model_predict(int N, int T, int TK, int S) {
    double smem_model = ((double)T * TK + TK) * 4.0 * (S + 1);   // v1.8.5 per-launch 口径（11-15 对齐）
    if (smem_model > 99 * 1024) return -1.0;
    long long numBlocks = ((long long)N + T - 1) / T;
    if (numBlocks < 64) return -1.0;                       // 半机填充守卫
    long long chunks = ((long long)N + TK - 1) / TK;
    double Lmin = (T < 16) ? 1260.0 : 820.0;
    double occ = std::min(1.0, (double)numBlocks / 128.0);
    double bwEff = 128.0 * 7.21 * occ * std::max(0.6, 1.0 - 2.4 / (double)T);
    double streamB = (double)N * N * 4.0;
    double pass = std::max((double)chunks * Lmin, streamB / bwEff)
                + (double)T * 4.0 / 7.21;                  // 输出写回
    return 2.0 * 3000.0 + 2.0 * pass;
}

// ---- 列 GEMV（pass1：out[col] += Σ_i r[i]·A[i][col]），cp.async (S+1) 槽 ----
// A 面板 TK×T k-major：元素 (r,c) = A[i0+r][j0+c]，行连续 16B 读合并；
// 线程 tid < T 持有列 j0+tid 的累加器（跨 chunk 存活）。
__global__ void gemv_cols_acc(const float *__restrict__ A, const float *__restrict__ rv,
                              float *__restrict__ out, int T, int TK, int S, int nSize) {
    extern __shared__ float slots[];
    int slotFloats = T * TK + TK;                 // [A-chunk TK×T | r-chunk TK]
    int tid = threadIdx.x, nth = blockDim.x;
    int j0 = blockIdx.x * T;
    int chunks = nSize / TK;                      // N % TK == 0（512 满足）

    auto prefetch = [&](int w) {
        int i0 = w * TK;
        float *dstA = slots + (w % (S + 1)) * slotFloats;
        float *dstR = dstA + T * TK;
        int ops = T / 4;                           // 列面板行宽 = T：每行（i）拷 T 列 = T/4 次 16B
        // 【11-15 修复】原抄行向 kernel 的 ops = TK/4 → q*4 越过 T 列边界踩踏面板
        //  （T=16/TK=16 巧合通过即此模式实证）+ 末列 tile 越界读 A → illegal access
        for (int k = tid; k < TK * ops; k += nth) {
            int r = k / ops, q = k % ops;
            __pipeline_memcpy_async(dstA + r * T + q * 4,
                                    A + (size_t)(i0 + r) * nSize + j0 + q * 4, 16);
        }
        for (int k = tid; k < TK / 4; k += nth)    // r-chunk 固定 TK float（与列面板行宽无关）
            __pipeline_memcpy_async(dstR + k * 4, rv + i0 + k * 4, 16);
        __pipeline_commit();
    };

    float acc = 0.f;
    int issued = 0;
    for (int k = 0; k <= S && k < chunks; ++k) { prefetch(k); ++issued; }
    for (int w = 0; w < chunks; ++w) {
        __pipeline_wait_prior((unsigned)(issued - w - 1));
        __syncthreads();
        const float *bA = slots + (w % (S + 1)) * slotFloats;
        const float *bR = bA + T * TK;
        if (tid < T) {
            for (int r = 0; r < TK; ++r) acc += bA[r * T + tid] * bR[r];
        }
        __syncthreads();
        int next = w + S + 1;
        if (next < chunks) { prefetch(next); ++issued; }
    }
    if (tid < T) out[j0 + tid] += acc;
}

// ---- 行 GEMV（pass2：out[row] += Σ_j A[row][j]×p[j]）——与 mvt_tiled 逐字节同构 ----
__global__ void gemv_rows_acc(const float *__restrict__ A, const float *__restrict__ y,
                              float *__restrict__ out, int T, int TK, int S, int nSize) {
    extern __shared__ float slots[];
    int slotFloats = T * TK + TK;                 // [A-chunk T×TK | y-chunk TK]
    int tid = threadIdx.x, nth = blockDim.x;
    int row0 = blockIdx.x * T;
    int chunks = nSize / TK;

    auto prefetch = [&](int w) {
        int j0 = w * TK;
        float *dstA = slots + (w % (S + 1)) * slotFloats;
        float *dstY = dstA + T * TK;
        int ops = TK / 4;
        for (int k = tid; k < T * ops; k += nth) {
            int r = k / ops, q = k % ops;
            __pipeline_memcpy_async(dstA + r * TK + q * 4,
                                    A + (size_t)(row0 + r) * nSize + j0 + q * 4, 16);
        }
        for (int k = tid; k < ops; k += nth)
            __pipeline_memcpy_async(dstY + k * 4, y + j0 + k * 4, 16);
        __pipeline_commit();
    };

    float acc = 0.f;
    int issued = 0;
    for (int k = 0; k <= S && k < chunks; ++k) { prefetch(k); ++issued; }
    for (int w = 0; w < chunks; ++w) {
        __pipeline_wait_prior((unsigned)(issued - w - 1));
        __syncthreads();
        const float *bA = slots + (w % (S + 1)) * slotFloats;
        const float *bY = bA + T * TK;
        if (tid < T) {
            const float *bAr = bA + tid * TK;
            for (int c = 0; c < TK; ++c) acc += bAr[c] * bY[c];
        }
        __syncthreads();
        int next = w + S + 1;
        if (next < chunks) { prefetch(next); ++issued; }
    }
    if (tid < T) out[row0 + tid] += acc;
}

int main(int argc, char** argv) {
    // 用法: ./bicg_tiled [N] [modelT] [modelTK] [modelS]
    int NN = 512, MT = -1, MTK = -1, MS = -1;
    int vals[4] = {0, 0, 0, 0}, nv = 0;
    for (int i = 1; i < argc && nv < 4; ++i) vals[nv++] = atoi(argv[i]);
    if (nv >= 1) NN = vals[0];
    if (nv >= 4) { MT = vals[1]; MTK = vals[2]; MS = vals[3]; }
    if (NN % 4 != 0) { printf("[参数] N 必须为 4 的倍数（16B 对齐），当前 %d\n", NN); return 1; }
    printf("[参数] N=%d, 模型标注: T=%d TK=%d S=%d\n", NN, MT, MTK, MS);

    size_t bA = (size_t)NN * NN * sizeof(float), bV = (size_t)NN * sizeof(float);
    float *hA = (float *)malloc(bA);
    float *hr = (float *)malloc(bV), *hp = (float *)malloc(bV);
    float *hs = (float *)malloc(bV), *hq = (float *)malloc(bV);
    float *rs = (float *)malloc(bV), *rq = (float *)malloc(bV);   // CPU double 参考
    for (int i = 0; i < NN * NN; ++i) hA[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < NN; ++i) {
        hr[i] = (float)(rand() % 100) / 100.0f;
        hp[i] = (float)(rand() % 100) / 100.0f;
        hs[i] = 0.f; hq[i] = 0.f;                  // 零初值（官方 bicg 语义）
    }
    // CPU double 参考：s[j] = Σ_i r[i]·A[i][j]；q[i] = Σ_j A[i][j]·p[j]
    for (int j = 0; j < NN; ++j) {
        double s = 0;
        for (int i = 0; i < NN; ++i) s += (double)hr[i] * hA[(size_t)i * NN + j];
        rs[j] = (float)s;
    }
    for (int i = 0; i < NN; ++i) {
        double q = 0;
        for (int j = 0; j < NN; ++j) q += (double)hA[(size_t)i * NN + j] * hp[j];
        rq[i] = (float)q;
    }

    float *dA, *dr, *dp, *ds, *dq;
    CHECK(cudaMalloc(&dA, bA)); CHECK(cudaMalloc(&dr, bV)); CHECK(cudaMalloc(&dp, bV));
    CHECK(cudaMalloc(&ds, bV)); CHECK(cudaMalloc(&dq, bV));
    CHECK(cudaMemcpy(dA, hA, bA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dr, hr, bV, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dp, hp, bV, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(ds, 0, bV));
    CHECK(cudaMemset(dq, 0, bV));
    CHECK(cudaFuncSetAttribute(gemv_cols_acc, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
    CHECK(cudaFuncSetAttribute(gemv_rows_acc, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));

    struct Row { int T, TK, S; double meas; bool modelOK; };
    std::vector<Row> rows;
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));

    printf("=== bicg 穷举对表 (N=%d, A=%.0fMB, pass1 列 GEMV + pass2 行 GEMV, 全空间实测) ===\n",
           NN, bA / 1048576.0);

    for (int T = 8; T <= 256; T *= 2)
    for (int TK = 16; TK <= 64; TK *= 2)
    for (int S = 1; S <= 4; ++S) {
        double pred = model_predict(NN, T, TK, S);
        bool modelOK = (pred >= 0);
        int tilesN = NN / T;
        int smem = (S + 1) * (T * TK + TK) * 4;
        if (smem > 101376) { printf("T=%-4d TK=%-3d S=%d  kernel smem=%5.1fKB 超限跳过\n",
                                    T, TK, S, smem / 1024.0); continue; }
        int grid = tilesN;

        // 正确性：清零 s/q → 单次 pass1+pass2 → 对 CPU double 参考（抽 500 点/pass）
        CHECK(cudaMemset(ds, 0, bV));
        CHECK(cudaMemset(dq, 0, bV));
        gemv_cols_acc<<<grid, 256, smem>>>(dA, dr, ds, T, TK, S, NN);
        gemv_rows_acc<<<grid, 256, smem>>>(dA, dp, dq, T, TK, S, NN);
        CHECK(cudaDeviceSynchronize());
        {
            cudaError_t e2 = cudaGetLastError();
            if (e2 != cudaSuccess) { printf("T=%-4d TK=%-3d S=%d  [fail] launch error: %s\n",
                                            T, TK, S, cudaGetErrorString(e2)); continue; }
        }
        CHECK(cudaMemcpy(hs, ds, bV, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hq, dq, bV, cudaMemcpyDeviceToHost));
        double maxRel = 0;
        for (int i = 0; i < NN; i += std::max(1, NN / 500)) {
            double r1 = fabs((double)hs[i] - (double)rs[i]) / (fabs((double)rs[i]) + 1e-6);
            double r2 = fabs((double)hq[i] - (double)rq[i]) / (fabs((double)rq[i]) + 1e-6);
            if (r1 > maxRel) maxRel = r1;
            if (r2 > maxRel) maxRel = r2;
        }
        bool ok = maxRel < 5e-3;
        if (!ok) { printf("T=%-4d TK=%-3d S=%d  正确性 FAIL (maxRel=%.2e) —— 跳过计时\n",
                          T, TK, S, maxRel); continue; }

        // 计时：3 rep × (pass1+pass2)，取最优（累加漂移不影响数据无关计时）
        double bestRep = 1e18;
        for (int rep = 0; rep < 3; ++rep) {
            float ms = 0;
            CHECK(cudaEventRecord(ev0));
            gemv_cols_acc<<<grid, 256, smem>>>(dA, dr, ds, T, TK, S, NN);
            gemv_rows_acc<<<grid, 256, smem>>>(dA, dp, dq, T, TK, S, NN);
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaEventSynchronize(ev1));
            CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            if (ms < bestRep) bestRep = ms;
        }
        double us = bestRep * 1000.0;
        printf("T=%-4d TK=%-3d S=%d  smem=%5.1fKB  实测: %12.1f us  模型: %s\n",
               T, TK, S, smem / 1024.0, us, modelOK ? "可行" : "拒(半机/smem)");
        rows.push_back({T, TK, S, us, modelOK});
    }

    // ---------------- 双排名 + 判定（与 mvt_tiled 同款） ----------------
    printf("\n===== 实测排名（升序，全空间 %d 配置）=====\n", (int)rows.size());
    std::vector<Row> sorted = rows;
    for (size_t i = 0; i < sorted.size(); ++i)
        for (size_t j = i + 1; j < sorted.size(); ++j)
            if (sorted[j].meas < sorted[i].meas) std::swap(sorted[i], sorted[j]);

    int modelRankFull = -1; double bestMeas = sorted[0].meas;
    for (size_t i = 0; i < sorted.size(); ++i) {
        bool isModel = (sorted[i].T == MT && sorted[i].TK == MTK && sorted[i].S == MS);
        if (isModel) modelRankFull = (int)i + 1;
        printf("%2zd   T=%-4d TK=%-3d S=%d   %12.1f us  %s%s\n", i + 1, sorted[i].T,
               sorted[i].TK, sorted[i].S, sorted[i].meas,
               sorted[i].modelOK ? "模型可行" : "模型拒",
               isModel ? "  <<<< 模型决策" : "");
    }

    std::vector<Row> feas;
    for (auto &r : rows) if (r.modelOK) feas.push_back(r);
    for (size_t i = 0; i < feas.size(); ++i)
        for (size_t j = i + 1; j < feas.size(); ++j)
            if (feas[j].meas < feas[i].meas) std::swap(feas[i], feas[j]);
    int modelRankFeas = -1; double bestFeas = feas[0].meas;
    for (size_t i = 0; i < feas.size(); ++i)
        if (feas[i].T == MT && feas[i].TK == MTK && feas[i].S == MS) modelRankFeas = (int)i + 1;

    if (modelRankFull > 0 && modelRankFeas > 0) {
        double modelMeas = 0;
        for (auto &r : sorted)
            if (r.T == MT && r.TK == MTK && r.S == MS) modelMeas = r.meas;
        double pctF = 100.0 * modelRankFeas / (double)feas.size();
        double gapF = 100.0 * (modelMeas - bestFeas) / bestFeas;
        double pctAll = 100.0 * modelRankFull / (double)sorted.size();
        double gapAll = 100.0 * (modelMeas - bestMeas) / bestMeas;
        printf("\n[判定] 模型决策 (T=%d,TK=%d,S=%d):\n", MT, MTK, MS);
        printf("  ① 模型可行子集: 穷举第 %d/%d 名 = 前 %.1f%%，距优 %+.1f%% → %s\n",
               modelRankFeas, (int)feas.size(), pctF, gapF,
               (pctF <= 20.0 && gapF <= 20.0) ? "前20% 达标" : "前20% 未达");
        printf("  ② 全空间:      穷举第 %d/%d 名 = 前 %.1f%%，距优 %+.1f%% → %s\n",
               modelRankFull, (int)sorted.size(), pctAll, gapAll,
               (pctAll <= 20.0 && gapAll <= 20.0) ? "前20% 达标" : "前20% 未达");
        if (modelRankFull == 1 && modelRankFeas == 1)
            printf("  [守卫检验] 模型决策 = 全空间最优 → 半机守卫未排除更优配置\n");
        else if (modelRankFull > 1 && sorted[0].modelOK == false)
            printf("  [守卫检验][重要] 全空间最优为模型拒配置 → 半机守卫证伪风险，必须登记！\n");
    } else {
        printf("\n[判定] 模型决策未在实测集中出现（标注参数或缺测）\n");
    }

    free(hA); free(hr); free(hp); free(hs); free(hq); free(rs); free(rq);
    CHECK(cudaFree(dA)); CHECK(cudaFree(dr)); CHECK(cudaFree(dp));
    CHECK(cudaFree(ds)); CHECK(cudaFree(dq));
    return 0;
}
