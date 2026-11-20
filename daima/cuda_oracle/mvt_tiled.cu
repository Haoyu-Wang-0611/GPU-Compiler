// mvt_tiled.cu —— mvt 组化链穷举对表 oracle（11-13，A3 主表第 10 行）
// 与 mvt_probe.mlir 同语义：pass1 x1[i] += A[i][j]·y1[j]；pass2 x2[i] += A[i][j]·y2[j]
//   两遍均为行主序 GEMV（与 atax 的"行+列转置"不同——mvt 只换 y 向量与输出）。
// 参数空间 = 模型候选空间全量：T ∈ {8..256} × TK ∈ {16,32,64} × S ∈ {1..4}（72 配置）
//
// 【协议改进（相对 atax_tiled）】双判据：atax_tiled 只测模型可行集（半机守卫
//   内），N=512 时可行集仅剩 T=8（12 配置）——排名太弱。本 oracle 全空间实测
//   （含模型拒的 T=16..256），判定行同时报：
//   ① 模型可行子集排名（与模型候选空间协议一致）；
//   ② 全空间排名（诚实性 + 半机守卫启发式的实测检验——若 T=16/32 实测更快，
//      守卫被证伪，必须登记）。
//
// 【预注册（11-13）】模型决策 (T=8, TK=64, S=1)，predicted 2.616888e+04 ns
//   （与 L1Feas v1.8 决策逐位同值）；判据：双排名均前 20% + 距优 ≤20%。
//   风险登记：T=8 时 kernel 256 线程仅 8 行活跃（248 空转）——线程效率损失
//   未建模，若实测差于此效应主导，登记为模型缺口。
//
// 编译/运行：
//   nvcc -O3 -arch=sm_89 mvt_tiled.cu -o mvt_tiled
//   CUDA_VISIBLE_DEVICES=1 ./mvt_tiled 512 8 64 1
//
// 语义注记：x1/x2 带初值累加（mvt 语义）；计时 rep 间累加漂移不影响数据无关
//   的访存计时；正确性单独用带初值的单次运行对 CPU double 参考。

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

// ---- L1Feas GEMV 两资源公式复现（P=2, M=K=N；v1.8 模型不变） ----
// 返回 -1 = 模型拒（smem 超限 / 半机守卫）
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

// ---- 行 GEMV（累加语义：out[row] += Σ_j A[row][j]×y[j]），cp.async (S+1) 槽 ----
// 与 atax_tiled 的 gemv_rows 同构，唯一差异：写回为 +=（mvt 累加语义）
__global__ void gemv_rows_acc(const float *__restrict__ A, const float *__restrict__ y,
                              float *__restrict__ out, int T, int TK, int S, int nSize) {
    extern __shared__ float slots[];
    int slotFloats = T * TK + TK;                 // [A-chunk T×TK | y-chunk TK]
    int tid = threadIdx.x, nth = blockDim.x;
    int row0 = blockIdx.x * T;
    int chunks = nSize / TK;                      // N % TK == 0（512 满足）

    auto prefetch = [&](int w) {
        int j0 = w * TK;
        float *dstA = slots + (w % (S + 1)) * slotFloats;
        float *dstY = dstA + T * TK;
        int ops = TK / 4;                          // 每行段 TK float = TK/4 次 16B
        for (int k = tid; k < T * ops; k += nth) {
            int r = k / ops, q = k % ops;
            __pipeline_memcpy_async(dstA + r * TK + q * 4,
                                    A + (size_t)(row0 + r) * nSize + j0 + q * 4, 16);
        }
        for (int k = tid; k < ops; k += nth)
            __pipeline_memcpy_async(dstY + k * 4, y + j0 + k * 4, 16);
        __pipeline_commit();
    };

    // 【11-10 修复继承】acc 跨 chunk 累加（线程 tid 持有行 row0+tid，T ≤ 256 = nth）
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
    if (tid < T) out[row0 + tid] += acc;           // mvt 累加语义（唯一差异）
}

int main(int argc, char** argv) {
    // 用法: ./mvt_tiled [N] [modelT] [modelTK] [modelS]
    int NN = 512, MT = -1, MTK = -1, MS = -1;
    int vals[4] = {0, 0, 0, 0}, nv = 0;
    for (int i = 1; i < argc && nv < 4; ++i) vals[nv++] = atoi(argv[i]);
    if (nv >= 1) NN = vals[0];
    if (nv >= 4) { MT = vals[1]; MTK = vals[2]; MS = vals[3]; }
    if (NN % 4 != 0) { printf("[参数] N 必须为 4 的倍数（16B 对齐），当前 %d\n", NN); return 1; }
    printf("[参数] N=%d, 模型标注: T=%d TK=%d S=%d\n", NN, MT, MTK, MS);

    size_t bA = (size_t)NN * NN * sizeof(float), bV = (size_t)NN * sizeof(float);
    float *hA = (float *)malloc(bA);
    float *hy1 = (float *)malloc(bV), *hy2 = (float *)malloc(bV);
    float *hx1 = (float *)malloc(bV), *hx2 = (float *)malloc(bV);
    float *rx1 = (float *)malloc(bV), *rx2 = (float *)malloc(bV);   // CPU double 参考
    for (int i = 0; i < NN * NN; ++i) hA[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < NN; ++i) {
        hy1[i] = (float)(rand() % 100) / 100.0f;
        hy2[i] = (float)(rand() % 100) / 100.0f;
        hx1[i] = (float)(1 + i % NN) * 0.01f;      // 带初值（mvt 语义）
        hx2[i] = (float)(1 + (i + 7) % NN) * 0.01f;
    }
    // CPU double 参考（两遍累加链）
    for (int i = 0; i < NN; ++i) {
        double s1 = (double)hx1[i], s2 = (double)hx2[i];
        for (int j = 0; j < NN; ++j) {
            s1 += (double)hA[(size_t)i * NN + j] * hy1[j];
            s2 += (double)hA[(size_t)i * NN + j] * hy2[j];
        }
        rx1[i] = (float)s1; rx2[i] = (float)s2;
    }

    float *dA, *dy1, *dy2, *dx1, *dx2, *dx1Init, *dx2Init;
    CHECK(cudaMalloc(&dA, bA)); CHECK(cudaMalloc(&dy1, bV)); CHECK(cudaMalloc(&dy2, bV));
    CHECK(cudaMalloc(&dx1, bV)); CHECK(cudaMalloc(&dx2, bV));
    CHECK(cudaMalloc(&dx1Init, bV)); CHECK(cudaMalloc(&dx2Init, bV));
    CHECK(cudaMemcpy(dA, hA, bA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dy1, hy1, bV, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dy2, hy2, bV, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx1, hx1, bV, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx2, hx2, bV, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx1Init, hx1, bV, cudaMemcpyHostToDevice));   // 初值快照（正确性用）
    CHECK(cudaMemcpy(dx2Init, hx2, bV, cudaMemcpyHostToDevice));
    CHECK(cudaFuncSetAttribute(gemv_rows_acc, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));

    struct Row { int T, TK, S; double meas; bool modelOK; };
    std::vector<Row> rows;
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));

    printf("=== mvt 穷举对表 (N=%d, A=%.0fMB, 2 pass 行 GEMV 累加链, 全空间实测) ===\n",
           NN, bA / 1048576.0);

    for (int T = 8; T <= 256; T *= 2)
    for (int TK = 16; TK <= 64; TK *= 2)
    for (int S = 1; S <= 4; ++S) {
        double pred = model_predict(NN, T, TK, S);
        bool modelOK = (pred >= 0);
        int tilesN = NN / T;                        // N % T == 0（512 满足）
        int smem = (S + 1) * (T * TK + TK) * 4;
        if (smem > 101376) { printf("T=%-4d TK=%-3d S=%d  kernel smem=%5.1fKB 超限跳过\n",
                                    T, TK, S, smem / 1024.0); continue; }
        int grid = tilesN;

        // 正确性：恢复初值 → 单次 pass1+pass2 → 对 CPU double 参考（抽 500 点/pass）
        CHECK(cudaMemcpy(dx1, dx1Init, bV, cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(dx2, dx2Init, bV, cudaMemcpyDeviceToDevice));
        gemv_rows_acc<<<grid, 256, smem>>>(dA, dy1, dx1, T, TK, S, NN);
        gemv_rows_acc<<<grid, 256, smem>>>(dA, dy2, dx2, T, TK, S, NN);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hx1, dx1, bV, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hx2, dx2, bV, cudaMemcpyDeviceToHost));
        double maxRel = 0;
        for (int i = 0; i < NN; i += std::max(1, NN / 500)) {
            double r1 = fabs((double)hx1[i] - (double)rx1[i]) / (fabs((double)rx1[i]) + 1e-6);
            double r2 = fabs((double)hx2[i] - (double)rx2[i]) / (fabs((double)rx2[i]) + 1e-6);
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
            gemv_rows_acc<<<grid, 256, smem>>>(dA, dy1, dx1, T, TK, S, NN);
            gemv_rows_acc<<<grid, 256, smem>>>(dA, dy2, dx2, T, TK, S, NN);
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

    // ---------------- 双排名 + 判定 ----------------
    printf("\n===== 实测排名（升序，全空间 %d 配置）=====\n", (int)rows.size());
    std::vector<Row> sorted = rows;
    for (size_t i = 0; i < sorted.size(); ++i)
        for (size_t j = i + 1; j < sorted.size(); ++j)
            if (sorted[j].meas < sorted[i].meas) std::swap(sorted[i], sorted[j]);

    // ① 全空间排名（含模型拒的配置）
    int modelRankFull = -1; double bestMeas = sorted[0].meas;
    for (size_t i = 0; i < sorted.size(); ++i) {
        bool isModel = (sorted[i].T == MT && sorted[i].TK == MTK && sorted[i].S == MS);
        if (isModel) modelRankFull = (int)i + 1;
        printf("%2zd   T=%-4d TK=%-3d S=%d   %12.1f us  %s%s\n", i + 1, sorted[i].T,
               sorted[i].TK, sorted[i].S, sorted[i].meas,
               sorted[i].modelOK ? "模型可行" : "模型拒",
               isModel ? "  <<<< 模型决策" : "");
    }

    // ② 模型可行子集排名（与模型候选空间协议一致）
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

    free(hA); free(hy1); free(hy2); free(hx1); free(hx2); free(rx1); free(rx2);
    CHECK(cudaFree(dA)); CHECK(cudaFree(dy1)); CHECK(cudaFree(dy2));
    CHECK(cudaFree(dx1)); CHECK(cudaFree(dx2));
    CHECK(cudaFree(dx1Init)); CHECK(cudaFree(dx2Init));
    return 0;
}
