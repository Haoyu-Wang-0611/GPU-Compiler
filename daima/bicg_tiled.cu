// bicg_tiled.cu —— 守卫形态之外的第一类"双朝向 GEMV 链" oracle（11-15）
//
// 语义（PolyBench bicg，与 bicg_probe.mlir 同构）：
//   pass1: s[j] = Σ_i r[i]·A[i][j]      ← 列向 GEMV（读 A 列带，归约 i）
//   pass2: q[i] = Σ_j A[i][j]·p[j]      ← 行向 GEMV（p = s）
// 与 atax 的唯一差别 = **两遍顺序相反**（atax 是先行后列）。载具直接复用
// atax_tiled.cu 的两个已验证 kernel（gemv_cols / gemv_rows，逐字节未改）。
//
// 与 atax_tiled 的两点不同（为了拿到"双判据"）：
//   ① 测**全空间**（只跳 smem 超限，不跳模型拒的配置）——这样既能报
//      "模型可行子集内名次"，也能报"全空间名次 + 守卫检验"（mvt 协议同款）；
//   ② 排名表同时给出两种口径的判定行。
//
// 用法:
//   ./bicg_tiled                      # N=512，无标注
//   ./bicg_tiled 512 8 64 1           # N + 模型标注 (T, TK, S)
//
// 编译:
//   nvcc -O3 -arch=sm_89 bicg_tiled.cu -o bicg_tiled

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

// 结构可行性（与 L1Feas v1.8.5 per-launch 口径一致）：只报"模型是否可行"，不跳测
static bool model_feasible(int N, int T, int TK, int S) {
    const double smemCap = 99 * 1024;
    const int numSms = 128;
    double smem_model = ((double)T * TK + TK) * 4 * (S + 1);
    if (smem_model > smemCap) return false;
    long long numBlocks = ((long long)N + T - 1) / T;
    if (numBlocks < numSms / 2) return false;
    return true;
}

// ---- 列向 GEMV（pass1）：out[col] = Σ_i A[i][col]×v[i] ----
__global__ void gemv_cols(const float *__restrict__ A, const float *__restrict__ v,
                          float *__restrict__ out, int T, int TK, int S, int nSize) {
    extern __shared__ float slots[];
    int slotFloats = TK * T + TK;
    int tid = threadIdx.x, nth = blockDim.x;
    int col0 = blockIdx.x * T;
    int chunks = nSize / TK;
    float acc = 0.f;

    auto prefetch = [&](int w) {
        int i0 = w * TK;
        float *dstA = slots + (w % (S + 1)) * slotFloats;
        float *dstV = dstA + TK * T;
        int ops = T / 4;
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
        if (tid < T) {
            for (int r = 0; r < TK; ++r) acc += bA[r * T + tid] * bV[r];
        }
        __syncthreads();
        int next = w + S + 1;
        if (next < chunks) { prefetch(next); ++issued; }
    }
    if (tid < T) out[col0 + tid] = acc;
}

// ---- 行向 GEMV（pass2）：out[row] = Σ_j A[row][j]×x[j] ----
__global__ void gemv_rows(const float *__restrict__ A, const float *__restrict__ x,
                          float *__restrict__ out, int T, int TK, int S, int nSize) {
    extern __shared__ float slots[];
    int slotFloats = T * TK + TK;
    int tid = threadIdx.x, nth = blockDim.x;
    int row0 = blockIdx.x * T;
    int chunks = nSize / TK;

    auto prefetch = [&](int w) {
        int j0 = w * TK;
        float *dstA = slots + (w % (S + 1)) * slotFloats;
        float *dstX = dstA + T * TK;
        int ops = TK / 4;
        for (int k = tid; k < T * ops; k += nth) {
            int r = k / ops, q = k % ops;
            __pipeline_memcpy_async(dstA + r * TK + q * 4,
                                    A + (size_t)(row0 + r) * nSize + j0 + q * 4, 16);
        }
        for (int k = tid; k < ops; k += nth)
            __pipeline_memcpy_async(dstX + k * 4, x + j0 + k * 4, 16);
        __pipeline_commit();
    };

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

int main(int argc, char** argv) {
    int NN = 512, MT = -1, MTK = -1, MS = -1;
    int vals[4] = {0, 0, 0, 0}, nv = 0;
    for (int i = 1; i < argc && nv < 4; ++i) vals[nv++] = atoi(argv[i]);
    if (nv >= 1) NN = vals[0];
    if (nv >= 4) { MT = vals[1]; MTK = vals[2]; MS = vals[3]; }
    if (NN % 256 != 0) { printf("[参数] N 必须被 256 整除（T 上限），当前 %d\n", NN); return 1; }
    printf("[参数] N=%d, 模型标注: T=%d TK=%d S=%d\n", NN, MT, MTK, MS);

    size_t bA = (size_t)NN * NN * sizeof(float), bV = (size_t)NN * sizeof(float);
    float *hA = (float *)malloc(bA), *hr = (float *)malloc(bV);
    float *hs = (float *)malloc(bV), *hq = (float *)malloc(bV);
    float *rs = (float *)malloc(bV), *rq = (float *)malloc(bV);
    for (int i = 0; i < NN * NN; ++i) hA[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < NN; ++i) hr[i] = (float)(rand() % 100) / 100.0f;

    // CPU double 参考：pass1 列向 → s，pass2 行向 → q
    for (int j = 0; j < NN; ++j) {
        double acc = 0;
        for (int i = 0; i < NN; ++i) acc += (double)hA[(size_t)i * NN + j] * hr[i];
        rs[j] = (float)acc;
    }
    for (int i = 0; i < NN; ++i) {
        double acc = 0;
        for (int j = 0; j < NN; ++j) acc += (double)hA[(size_t)i * NN + j] * rs[j];
        rq[i] = (float)acc;
    }

    float *dA, *dr, *ds, *dq;
    CHECK(cudaMalloc(&dA, bA)); CHECK(cudaMalloc(&dr, bV));
    CHECK(cudaMalloc(&ds, bV)); CHECK(cudaMalloc(&dq, bV));
    CHECK(cudaMemcpy(dA, hA, bA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dr, hr, bV, cudaMemcpyHostToDevice));
    CHECK(cudaFuncSetAttribute(gemv_cols, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
    CHECK(cudaFuncSetAttribute(gemv_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));

    struct Row { int T, TK, S; double meas; bool feas; };
    std::vector<Row> rows;
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));

    printf("=== bicg 穷举对表 (N=%d, 2 pass 列向+行向 GEMV, 全空间实测) ===\n", NN);
    printf("  候选空间: T∈{8..256} × TK∈{16,32,64} × S∈{1..4}；模型拒的配置照测（双判据）\n\n");

    for (int T = 8; T <= 256; T *= 2)
    for (int TK = 16; TK <= 64; TK *= 2)
    for (int S = 1; S <= 4; ++S) {
        bool feas = model_feasible(NN, T, TK, S);
        int smem = (S + 1) * (T * TK + TK) * 4;
        if (smem > 101376) {
            printf("T=%-4d TK=%-3d S=%d  smem=%5.1fKB 超限跳过（模型亦拒）\n",
                   T, TK, S, smem / 1024.0);
            continue;
        }
        int grid = NN / T;

        // 正确性：pass1 列向 → ds；pass2 行向 → dq
        CHECK(cudaMemset(ds, 0, bV)); CHECK(cudaMemset(dq, 0, bV));
        gemv_cols<<<grid, 256, smem>>>(dA, dr, ds, T, TK, S, NN);
        CHECK(cudaDeviceSynchronize());
        gemv_rows<<<grid, 256, smem>>>(dA, ds, dq, T, TK, S, NN);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hs, ds, bV, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hq, dq, bV, cudaMemcpyDeviceToHost));
        double maxRel = 0;
        int step = std::max(1, NN / 500);
        for (int i = 0; i < NN; i += step) {
            double r1 = fabs((double)hs[i] - (double)rs[i]) / (fabs((double)rs[i]) + 1e-6);
            double r2 = fabs((double)hq[i] - (double)rq[i]) / (fabs((double)rq[i]) + 1e-6);
            if (r1 > maxRel) maxRel = r1;
            if (r2 > maxRel) maxRel = r2;
        }
        bool ok = maxRel < 5e-3;

        double bestRep = 1e18;
        for (int rep = 0; rep < 3; ++rep) {
            float ms = 0;
            CHECK(cudaEventRecord(ev0));
            gemv_cols<<<grid, 256, smem>>>(dA, dr, ds, T, TK, S, NN);
            gemv_rows<<<grid, 256, smem>>>(dA, ds, dq, T, TK, S, NN);
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaEventSynchronize(ev1));
            CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            if (ms < bestRep) bestRep = ms;
        }
        double us = bestRep * 1000.0;
        printf("T=%-4d TK=%-3d S=%d  smem=%5.1fKB  实测: %9.1f us  %s  模型:%s\n",
               T, TK, S, smem / 1024.0, us, ok ? "PASS" : "FAIL",
               feas ? "可行" : "拒(半机/smem)");
        rows.push_back({T, TK, S, us, feas});
    }

    // ---------------- 双判据排名 ----------------
    std::vector<Row> sorted = rows;
    for (size_t i = 0; i < sorted.size(); ++i)
        for (size_t j = i + 1; j < sorted.size(); ++j)
            if (sorted[j].meas < sorted[i].meas) std::swap(sorted[i], sorted[j]);

    printf("\n===== 实测排名（全空间 %d 配置，升序）=====\n", (int)sorted.size());
    int mRank = -1;
    for (size_t i = 0; i < sorted.size(); ++i) {
        bool isM = (sorted[i].T == MT && sorted[i].TK == MTK && sorted[i].S == MS);
        if (isM) mRank = (int)i + 1;
        printf("%3zd  T=%-4d TK=%-3d S=%d  %9.1f us  %s%s\n", i + 1, sorted[i].T,
               sorted[i].TK, sorted[i].S, sorted[i].meas,
               sorted[i].feas ? "模型可行" : "模型拒", isM ? "  <<<< 模型决策" : "");
    }

    std::vector<Row> feas = rows;
    std::vector<Row> fv;
    for (auto &r : feas) if (r.feas) fv.push_back(r);
    for (size_t i = 0; i < fv.size(); ++i)
        for (size_t j = i + 1; j < fv.size(); ++j)
            if (fv[j].meas < fv[i].meas) std::swap(fv[i], fv[j]);
    int mRankF = -1;
    for (size_t i = 0; i < fv.size(); ++i)
        if (fv[i].T == MT && fv[i].TK == MTK && fv[i].S == MS) mRankF = (int)i + 1;

    if (mRank > 0 && mRankF > 0) {
        double mMeas = 0;
        for (auto &r : rows)
            if (r.T == MT && r.TK == MTK && r.S == MS) mMeas = r.meas;
        double gAll = 100.0 * (mMeas - sorted[0].meas) / sorted[0].meas;
        double gF = 100.0 * (mMeas - fv[0].meas) / fv[0].meas;
        printf("\n[判定] 模型决策 (T=%d,TK=%d,S=%d):\n", MT, MTK, MS);
        printf("  ① 可行子集: 第 %d/%d 名 = 前 %.1f%%，距优 %+.1f%% → %s\n",
               mRankF, (int)fv.size(), 100.0 * mRankF / fv.size(), gF,
               (100.0 * mRankF / fv.size() <= 20 && gF <= 20) ? "前20% 达标" : "前20% 未达");
        printf("  ② 全空间  : 第 %d/%d 名 = 前 %.1f%%，距优 %+.1f%% → %s\n",
               mRank, (int)sorted.size(), 100.0 * mRank / sorted.size(), gAll,
               (100.0 * mRank / sorted.size() <= 20 && gAll <= 20) ? "前20% 达标" : "前20% 未达");
        if (mRankF == 1) printf("  [守卫检验] 可行子集最优 = 模型决策\n");
        else if (!sorted[0].feas) printf("  [守卫检验][重要] 全空间最优是模型拒的配置 → 守卫证伪风险\n");
    } else {
        printf("\n[判定] 模型决策未出现在实测集中（标注参数或缺测）\n");
    }

    free(hA); free(hr); free(hs); free(hq); free(rs); free(rq);
    CHECK(cudaFree(dA)); CHECK(cudaFree(dr)); CHECK(cudaFree(ds)); CHECK(cudaFree(dq));
    return 0;
}
