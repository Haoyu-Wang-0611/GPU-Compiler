// heat1d_oracle.cu — heat1d 大尺寸穷举对表 oracle（11-14，D1 收尾）
// 载体 = cp_async_heat1d 的占位直通 kernel（流量 = heat1d 8B/元素：读4B+写4B），
//   T/S 改运行时参数（单实例；div/mod 开销相对 tile 工作量可忽略，排名保真）。
//
// 【占位载具 caveat】无 halo（相邻 tile 的 i±1 边界由 L2 吸收，二阶效应）——
//   与校准载具同一语义（l2_model 流量模型），登记。
//
// 【协议 = mvt 双判据】全空间实测（含模型拒的大 T），判定行同时报：
//   ① 模型可行子集排名（老 1D 模板的 numTiles≥512 守卫，N=2²⁰ → T≤2048，36 配置）；
//   ② 全空间排名（含守卫拒的 T=4096..2²⁰）+ 守卫检验（大 T 若更快 → 守卫证伪登记）。
//
// 【预注册（11-14）】模型决策 (Tile=2048, S=1)，predicted 1.234105e+04 ns；
//   判据：双排名均前 20% + 距优 ≤20%；占位流量 8 MB → 实测预期 ~10–25 µs 区间。
//
// 编译/运行：
//   nvcc -O3 -arch=sm_89 heat1d_oracle.cu -o heat1d_oracle
//   CUDA_VISIBLE_DEVICES=1 ./heat1d_oracle 1048576 2048 1

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define NUM_SMS 128
#define SMEM_CAP 101376

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            printf("CUDA error %s @%d\n", cudaGetErrorString(e), __LINE__); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// ---- 占位直通 kernel（与 cp_async_heat1d.cu 逐字同源，T/S 运行时化） ----
__global__ void heat1d_async(const float* __restrict__ in,
                             float* __restrict__ out, int n, int T, int S) {
    extern __shared__ float smem[];              // (S+1) * T floats
    const int NTILE = (n + T - 1) / T;
    const int G = gridDim.x;

    for (int j = 0; j < S; ++j) {
        int k = blockIdx.x + j * G;
        if (k >= NTILE) break;
        int base = k * T;
        for (int i = threadIdx.x; i < T; i += blockDim.x) {
            int g = base + i;
            if (g < n)
                __pipeline_memcpy_async(&smem[j * T + i], &in[g], 4);
        }
        __pipeline_commit();
    }

    int my_tiles = (NTILE - blockIdx.x + G - 1) / G;
    for (int c = 0; c < my_tiles; ++c) {
        int k = blockIdx.x + c * G;
        int slot = c % (S + 1);
        __pipeline_wait_prior(S - 1);
        __syncthreads();

        int base = k * T;
        for (int i = threadIdx.x; i < T; i += blockDim.x) {
            int g = base + i;
            if (g < n) out[g] = smem[slot * T + i];   // 占位计算（流量=heat1d）
        }
        __syncthreads();                              // slot 读完才允许覆写

        int kf = k + S * G;
        if (kf < NTILE) {
            int fslot = (c + S) % (S + 1);
            int fbase = kf * T;
            for (int i = threadIdx.x; i < T; i += blockDim.x) {
                int g = fbase + i;
                if (g < n)
                    __pipeline_memcpy_async(&smem[fslot * T + i], &in[g], 4);
            }
        }
        __pipeline_commit();
    }
}

// ---- 正确性：占位直通，out 必须等于 in ----
__global__ void check_kernel(const float* in, const float* out, int n, int* err) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && out[i] != in[i]) atomicAdd(err, 1);
}

// ---- 老 1D 模板的可行性复制（仅结构守卫，D3 纪律：不复制时间公式）----
// 守卫 = numTiles ≥ 512（N ≥ 8192 时）+ smem 上限；时间公式不复制（预测列已废）
static bool model_feasible(int N, int T, int S) {
    long long numTiles = ((long long)N + T - 1) / T;
    if (N >= 8192 && numTiles < 512) return false;   // 波次量化守卫（老模板原样）
    if ((long long)(S + 1) * T * 4 > 99 * 1024) return false;
    return true;
}

int main(int argc, char** argv) {
    // 用法: ./heat1d_oracle [N] [modelT] [modelS]
    long long N = 1 << 20;
    int MT = -1, MS = -1;
    long long v[3] = {0, 0, 0};
    int nv = 0;
    for (int i = 1; i < argc && nv < 3; ++i) v[nv++] = atoll(argv[i]);
    if (nv >= 1) N = v[0];
    if (nv >= 3) { MT = (int)v[1]; MS = (int)v[2]; }
    printf("[参数] N=%lld, 模型标注: Tile=%d S=%d\n", N, MT, MS);
    if (N % 4 != 0) { printf("[参数] N 必须为 4 的倍数（16B 对齐）\n"); return 1; }

    size_t bytes = (size_t)N * sizeof(float);
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes)); CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemset(d_in, 1, bytes));
    int* d_err; CHECK(cudaMalloc(&d_err, 4));

    // 【11-14 保险修复】>48KB 动态 smem 必须显式 opt-in，否则 launch 静默失败
    //   → 事件对只测到失败空转（首轮 2.2 µs 假数据根因，DeepSeek 式教训：
    //   sgemm_tiled 的"launch 失败必须拦截"保险未继承）。
    CHECK(cudaFuncSetAttribute(heat1d_async,
          cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_CAP));

    struct Row { int T, S; double us; bool modelOK; };
    std::vector<Row> rows;
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));

    printf("=== heat1d 穷举对表 (N=%lld, %.0fMB, 占位直通 8B/元素, 全空间实测) ===\n",
           N, bytes / 1048576.0);

    for (long long T = 16; T <= N; T *= 2)
    for (int S = 1; S <= 4; ++S) {
        bool modelOK = model_feasible((int)N, (int)T, S);
        long long smem = (S + 1) * T * 4;
        if (smem > SMEM_CAP) {
            printf("T=%-8lld S=%d  smem=%6.1fMB 超限跳过  模型: %s\n",
                   T, S, smem / 1048576.0, modelOK ? "可行" : "拒");
            continue;
        }
        int grid = NUM_SMS;

        // 正确性：直通校验（每个配置一次）——launch 错误必须拦截（【保险】同款）
        heat1d_async<<<grid, 256, smem>>>(d_in, d_out, (int)N, (int)T, S);
        { cudaError_t e = cudaGetLastError();
          if (e != cudaSuccess) { printf("T=%-8lld S=%d  launch FAIL: %s —— 跳过\n",
                                         T, S, cudaGetErrorString(e)); continue; } }
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(d_err, 0, 4));
        check_kernel<<<(int)((N + 255) / 256), 256>>>(d_in, d_out, (int)N, d_err);
        int h_err = -1;
        CHECK(cudaMemcpy(&h_err, d_err, 4, cudaMemcpyDeviceToHost));
        if (h_err != 0) { printf("T=%-8lld S=%d  正确性 FAIL (mismatch=%d)\n", T, S, h_err); continue; }

        // 计时：3 rep 取最优（单发射/rep）
        double bestRep = 1e18;
        for (int rep = 0; rep < 3; ++rep) {
            float ms = 0;
            CHECK(cudaEventRecord(ev0));
            heat1d_async<<<grid, 256, smem>>>(d_in, d_out, (int)N, (int)T, S);
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaEventSynchronize(ev1));
            CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            if (ms < bestRep) bestRep = ms;
        }
        { cudaError_t e = cudaGetLastError();
          if (e != cudaSuccess) { printf("T=%-8lld S=%d  计时 launch FAIL: %s —— 跳过\n",
                                         T, S, cudaGetErrorString(e)); continue; } }
        double us = bestRep * 1000.0;
        printf("T=%-8lld S=%d  smem=%5.1fKB  实测: %12.1f us  模型: %s\n",
               T, S, smem / 1024.0, us, modelOK ? "可行" : "拒(numTiles)");
        rows.push_back({(int)T, S, us, modelOK});
    }

    // ---------------- 双排名 + 判定 ----------------
    printf("\n===== 实测排名（升序，全空间 %d 配置）=====\n", (int)rows.size());
    std::vector<Row> sorted = rows;
    for (size_t i = 0; i < sorted.size(); ++i)
        for (size_t j = i + 1; j < sorted.size(); ++j)
            if (sorted[j].us < sorted[i].us) std::swap(sorted[i], sorted[j]);

    int modelRankFull = -1; double bestMeas = sorted[0].us;
    for (size_t i = 0; i < sorted.size(); ++i) {
        bool isModel = (sorted[i].T == MT && sorted[i].S == MS);
        if (isModel) modelRankFull = (int)i + 1;
        printf("%2zd   T=%-8d S=%d   %12.1f us  %s%s\n", i + 1, sorted[i].T,
               sorted[i].S, sorted[i].us,
               sorted[i].modelOK ? "模型可行" : "模型拒",
               isModel ? "  <<<< 模型决策" : "");
    }

    std::vector<Row> feas;
    for (auto &r : rows) if (r.modelOK) feas.push_back(r);
    for (size_t i = 0; i < feas.size(); ++i)
        for (size_t j = i + 1; j < feas.size(); ++j)
            if (feas[j].us < feas[i].us) std::swap(feas[i], feas[j]);
    int modelRankFeas = -1; double bestFeas = feas[0].us;
    for (size_t i = 0; i < feas.size(); ++i)
        if (feas[i].T == MT && feas[i].S == MS) modelRankFeas = (int)i + 1;

    if (modelRankFull > 0 && modelRankFeas > 0) {
        double modelMeas = 0;
        for (auto &r : sorted)
            if (r.T == MT && r.S == MS) modelMeas = r.us;
        double pctF = 100.0 * modelRankFeas / (double)feas.size();
        double gapF = 100.0 * (modelMeas - bestFeas) / bestFeas;
        double pctAll = 100.0 * modelRankFull / (double)sorted.size();
        double gapAll = 100.0 * (modelMeas - bestMeas) / bestMeas;
        printf("\n[判定] 模型决策 (Tile=%d,S=%d):\n", MT, MS);
        printf("  ① 模型可行子集: 穷举第 %d/%d 名 = 前 %.1f%%，距优 %+.1f%% → %s\n",
               modelRankFeas, (int)feas.size(), pctF, gapF,
               (pctF <= 20.0 && gapF <= 20.0) ? "前20% 达标" : "前20% 未达");
        printf("  ② 全空间:      穷举第 %d/%d 名 = 前 %.1f%%，距优 %+.1f%% → %s\n",
               modelRankFull, (int)sorted.size(), pctAll, gapAll,
               (pctAll <= 20.0 && gapAll <= 20.0) ? "前20% 达标" : "前20% 未达");
        if (modelRankFull == 1)
            printf("  [守卫检验] 模型决策 = 全空间最优 → numTiles≥512 守卫未排除更优配置\n");
        else if (!sorted[0].modelOK)
            printf("  [守卫检验][重要] 全空间最优为模型拒配置 → 守卫证伪风险，必须登记！\n");
    } else {
        printf("\n[判定] 模型决策未在实测集中出现（标注参数或缺测）\n");
    }

    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out)); CHECK(cudaFree(d_err));
    return 0;
}
