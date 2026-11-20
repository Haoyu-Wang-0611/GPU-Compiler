#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include <memory>
#include <string>
#include <optional>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace mlir;
using namespace mlir::affine;

namespace {

// 把 affine 表达式拆成 (dim 位置, 常数偏移)：只支持 dim 或 dim+c；
// dim+dim、除法、取模 → 返回 false（v1 不支持，走 fallback）
static bool splitDimConst(AffineExpr e, int &dimPos, int &c) {
  if (auto ce = dyn_cast<AffineConstantExpr>(e)) {
    dimPos = -1; c = (int)ce.getValue(); return true;
  }
  if (auto de = dyn_cast<AffineDimExpr>(e)) {
    dimPos = (int)de.getPosition(); c = 0; return true;
  }
  if (auto be = dyn_cast<AffineBinaryOpExpr>(e)) {
    if (be.getKind() != AffineExprKind::Add) return false;
    int d1, c1, d2, c2;
    if (!splitDimConst(be.getLHS(), d1, c1) || !splitDimConst(be.getRHS(), d2, c2)) return false;
    if (d1 >= 0 && d2 >= 0) return false;          // dim+dim：复杂，v1 放弃
    dimPos = (d1 >= 0) ? d1 : d2; c = c1 + c2; return true;
  }
  return false;
}

struct L1FeasPass
    : public PassWrapper<L1FeasPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(L1FeasPass)

  Option<int> numStages{*this, "num-stages", llvm::cl::desc("software pipeline depth"), llvm::cl::init(3)};
  Option<int> tileSize{*this, "tile-size", llvm::cl::desc("tile size for the loop"), llvm::cl::init(64)};

  L1FeasPass() = default;
  L1FeasPass(const L1FeasPass &other) : PassWrapper(other) {
    numStages = other.numStages;
    tileSize = other.tileSize;
  }

  StringRef getArgument() const override { return "l1-feasibility"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // ---- 窗口法 A：嵌套 for 外→内编号，建 IV 查找表 ----
    SmallVector<AffineForOp, 4> nestLoops;
    func->walk([&](AffineForOp forOp) { nestLoops.push_back(forOp); });
    // 实测（11-01）：本构建的 walk 把最内层循环排在最前 → 反转为外→内
    std::reverse(nestLoops.begin(), nestLoops.end());
    DenseMap<Value, int> ivIndex;
    for (unsigned i = 0; i < nestLoops.size(); ++i)
      ivIndex[nestLoops[i].getInductionVar()] = (int)i;

    // ---- 窗口法 B：访问描述 ----
    // dims[memref维号] = [(循环序号, 常数偏移c), ...]（同 memref 全部 op 合并）
    struct AccessDesc {
      Value memref;
      int elemBytes;
      bool isWritten = false;   // 该 memref 是否被 store（读/写区分 → 归约维探测）
      bool isRead = false;      // 该 memref 是否被 load（v1.2.2：只有被读的窗口进 smem）
      SmallVector<SmallVector<std::pair<int,int>, 2>> dims;
    };
    SmallVector<AccessDesc> winAccesses;   // 窗口法收集器（per-memref 合并视图）
    SmallVector<AccessDesc> winPanels;     // v1.8.3：per-访问面板视图（每个 load/store 一条；
                                           //   单 GEMM 分支双面板记账用，其余分支不读它）
    bool windowFallback = false;           // 出现复杂表达式 → 整体走 fallback
    DenseSet<int> readLoopSet, writeLoopSet;  // 归约维探测：读/写循环集合

    // ---- 窗口法 C：遍历 affine load/store，提取访问描述 ----
    func->walk([&](Operation *op) {
      AffineMap amap;
      Value memref;
      SmallVector<Value, 4> mapOps;
      bool isStore = false;
      if (auto ld = dyn_cast<affine::AffineLoadOp>(op)) {
        amap = ld.getAffineMap(); memref = ld.getMemRef();
        for (Value v : ld.getMapOperands()) mapOps.push_back(v);
      } else if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
        amap = st.getAffineMap(); memref = st.getMemRef();
        for (Value v : st.getMapOperands()) mapOps.push_back(v);
        isStore = true;
      } else {
        return WalkResult::advance();
      }

      // 找到/创建该 memref 的 AccessDesc（用下标而非指针，防 push_back 悬垂）
      size_t idx = winAccesses.size();
      for (size_t a = 0; a < winAccesses.size(); ++a)
        if (winAccesses[a].memref == memref) { idx = a; break; }
      if (idx == winAccesses.size()) {
        AccessDesc ad;
        ad.memref = memref;
        ad.elemBytes = cast<MemRefType>(memref.getType()).getElementTypeBitWidth() / 8;
        ad.dims.resize(amap.getNumResults());
        winAccesses.push_back(std::move(ad));
      }
      AccessDesc &acc = winAccesses[idx];
      if (isStore) acc.isWritten = true;
      else acc.isRead = true;

      // v1.8.3：per-访问面板描述（每 op 一条；dims[d] 至多 1 entry——单条仿射
      // 结果式只分解出一对 (dimPos, c)）。合并进 winAccesses 的行为不变。
      AccessDesc opPanel;
      opPanel.memref = memref;
      opPanel.elemBytes = cast<MemRefType>(memref.getType()).getElementTypeBitWidth() / 8;
      opPanel.dims.resize(amap.getNumResults());
      opPanel.isRead = !isStore;
      opPanel.isWritten = isStore;

      // 逐维分解
      for (unsigned d = 0; d < amap.getNumResults(); ++d) {
        int dimPos, c;
        if (!splitDimConst(amap.getResult(d), dimPos, c)) {
          windowFallback = true;                   // 复杂表达式（除法/取模/dim+dim）
          return WalkResult::advance();
        }
        if (dimPos < 0) continue;                  // 纯常数下标：该维不随循环变 → 窗口×1，不记
        Value iv = mapOps[dimPos];                 // 下标表达式第 dimPos 维的操作数
        auto it = ivIndex.find(iv);
        if (it == ivIndex.end()) continue;         // 函数实参等非循环 IV → 该维当作常数处理
        acc.dims[d].push_back({it->second, c});
        opPanel.dims[d].push_back({it->second, c});
        (isStore ? writeLoopSet : readLoopSet).insert(it->second);
      }
      winPanels.push_back(std::move(opPanel));
      return WalkResult::advance();
    });

    // ---- 窗口法调试打印：人工核对访问描述 ----
    llvm::errs() << "[win] collected " << winAccesses.size() << " memrefs"
                 << (windowFallback ? "  [WARNING: complex expr -> fallback]" : "") << "\n";
    for (unsigned i = 0; i < nestLoops.size(); ++i)
      llvm::errs() << "[win] L" << i << " loc = " << nestLoops[i].getLoc() << "\n";
    for (auto &a : winAccesses) {
      llvm::errs() << "[win] memref : ";
      for (unsigned d = 0; d < a.dims.size(); ++d) {
        llvm::errs() << "dim" << d << "={";
        for (auto &pr : a.dims[d])
          llvm::errs() << "(L" << pr.first << "," << pr.second << ")";
        llvm::errs() << "} ";
      }
      llvm::errs() << "\n";
    }

    // ---- 窗口法 footprint：window_维 = T(携带循环) + (max_c − min_c)，无依赖维 ×1 ----
    bool multiTileWarned = false;   // v1.8.2（D4）：同维多载体异值告警，每函数一次
    auto windowFootprint = [&](const AccessDesc &a, ArrayRef<int> loopTiles) -> int64_t {
      auto memrefType = cast<MemRefType>(a.memref.getType());
      int64_t bytes = a.elemBytes;
      for (size_t d = 0; d < a.dims.size(); ++d) {
        if (a.dims[d].empty()) continue;            // 该维不随任何循环变 → 数据复用，×1
        // v1.8.2（11-14，D4）：同维多环携带 → tile 取各携带环的最大值（保守）。
        //   修复 last-wins 漏洞（DeepSeek #3）：大/小块混用时窗口被低估 →
        //   误判可行 → 威胁零非法招牌。单载体/等值载体行为不变（锚点安全）。
        int Tmax = -1, Tlast = -1;
        int lo = a.dims[d].front().second, hi = a.dims[d].front().second;
        bool ok = true;
        for (auto &pr : a.dims[d]) {
          if (pr.first < 0 || pr.first >= (int)loopTiles.size()) { ok = false; break; }
          Tlast = loopTiles[pr.first];
          Tmax = std::max(Tmax, Tlast);
          lo = std::min(lo, pr.second);
          hi = std::max(hi, pr.second);
        }
        if (!ok) return -1;
        if (Tmax != Tlast && !multiTileWarned) {
          llvm::errs() << "[L1 Win] 注意(v1.8.2/D4): 同一 memref 维被多个不同 tile 的环携带"
                       << "——窗口按最大 tile 计（保守，防 last-wins 低估）\n";
          multiTileWarned = true;
        }
        int64_t window = (int64_t)Tmax + (hi - lo);    // halo 自动并入（heat1d 的 ±1 → +2）
        int64_t dimCap = memrefType.getDimSize(d);  // 不超过 memref 本维大小
        bytes *= std::min(window, dimCap);
      }
      return bytes;
    };

    // ---- 第 2 步验收：统一 tile=tileSize 的窗口打印 ----
    {
      SmallVector<int, 4> loopTiles(nestLoops.size(), tileSize);
      llvm::errs() << "[L1 Win] window footprints (uniform T=" << tileSize << "):\n";
      int64_t winTotal = 0; bool bad = false;
      for (auto &a : winAccesses) {
        int64_t fp = windowFootprint(a, loopTiles);
        if (fp < 0) { bad = true; continue; }
        winTotal += fp;
        llvm::errs() << "[L1 Win] memref window = " << fp << " B\n";
      }
      if (!bad) {
        int64_t winRequired = winTotal * (numStages + 1);
        llvm::errs() << "[L1 Win] smem required (conservative, C 也多缓冲) = " << winRequired
                     << " B -> " << (winRequired <= 99 * 1024 ? "FEASIBLE" : "INFEASIBLE") << "\n";
      }
    }

    // ==========================================
    // 【特征提取 1】FLOPs 计数 (逐 op 归属)
    // ==========================================
    int64_t arithCount = 0;
    int64_t dynamicFlops = 0;
    bool hasNonConst = false;
    func->walk([&](Operation *op) {
      if (!isa<arith::AddFOp, arith::MulFOp>(op)) return;
      arithCount++;
      int64_t dyn = 1;
      for (Operation *p = op->getParentOp(); p != nullptr && !isa<func::FuncOp>(p); p = p->getParentOp()) {
        if (auto forOp = dyn_cast<AffineForOp>(p)) {
          auto trip = forOp.getStaticTripCount();
          if (!trip.has_value()) { hasNonConst = true; return; }
          dyn *= static_cast<int64_t>(trip->getZExtValue());
        }
      }
      dynamicFlops += dyn;
    });

    int64_t totalTrip = 1;
    func->walk([&](AffineForOp forOp) {
      auto trip = forOp.getStaticTripCount();
      if (trip.has_value()) totalTrip *= static_cast<int64_t>(trip->getZExtValue());
      else hasNonConst = true;
    });

    double flops_per_iter = (totalTrip > 0) ? (double)dynamicFlops / (double)totalTrip : 0.0;
    if (arithCount > 0) {
      llvm::errs() << "[L1] FLOPs/iter = " << flops_per_iter << ", total trip = " << totalTrip << "\n";
    }

    // ==========================================
    // 【特征提取 2】收集 MemRef footprint（旧 1D 语义，保留作 fallback）
    // ==========================================
    DenseMap<Value, std::unique_ptr<MemRefRegion>> regions;
    func->walk([&](Operation *op) {
      Value memref;
      if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) memref = loadOp.getMemRef();
      else if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) memref = storeOp.getMemRef();
      else return WalkResult::advance();

      unsigned loopDepth = 1;
      MemRefRegion region(op->getLoc());
      if (failed(region.compute(op, loopDepth))) return WalkResult::advance();

      auto it = regions.find(memref);
      if (it == regions.end()) regions[memref] = std::make_unique<MemRefRegion>(region);
      else (void)it->second->unionBoundingBox(region);
      return WalkResult::advance();
    });

    std::vector<std::pair<int64_t, int64_t>> memrefs;
    int64_t N = 0;

    // 【回归锚点保留】计算固定 tileSize 下的 footprint
    int64_t anchorTotalBytes = 0;

    // v1.8.4：Anchor 行确定性排序——regions 为 DenseMap，迭代序跨运行不稳定
    //（11-13 金样化自检实证：行内容相同仅顺序漂移）。按 memref 类型名排序后
    // 打印；行格式不变（纯重排，regress.sh 排序 diff 透明，无需 rebase）。
    SmallVector<std::pair<std::string, int64_t>, 8> anchorLines;
    for (auto &[value, region] : regions) {
      std::optional<int64_t> baseBytes = region->getRegionSize();
      if (!baseBytes) continue;
      auto memrefType = cast<MemRefType>(value.getType());
      int64_t elemBytes = memrefType.getElementTypeBitWidth() / 8;

      memrefs.push_back({*baseBytes, elemBytes});
      if (N == 0 && memrefType.getRank() > 0)
        N = memrefType.getDimSize(0);

      int64_t anchorFootprint = *baseBytes + (tileSize - 1) * elemBytes;
      anchorTotalBytes += anchorFootprint;
      std::string ty;
      llvm::raw_string_ostream rs(ty);
      memrefType.print(rs);
      anchorLines.emplace_back(std::move(ty), anchorFootprint);
    }
    std::sort(anchorLines.begin(), anchorLines.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    for (auto &al : anchorLines)
      llvm::errs() << "[L1 Anchor] memref footprint (T=" << tileSize << ") = " << al.second << " B\n";

    if (memrefs.empty() || N == 0) {
      llvm::errs() << "[L1] No valid memrefs found, skipping decision.\n";
      return;
    }

    int64_t anchorRequired = anchorTotalBytes * (numStages + 1);
    llvm::errs() << "[L1 Anchor] smem required = " << anchorRequired << " B -> "
                 << (anchorRequired <= 99 * 1024 ? "FEASIBLE" : "INFEASIBLE") << "\n";

    // ==========================================
    // 【特征提取 3】kernel-class 探测（方案 B：IR 结构事实）+ 模板分发
    // 归约维 = IV 出现在被读 memref 下标、不出现在被写 memref 下标
    // ==========================================
    SmallVector<int, 4> reductionLoops, outputLoops;
    for (unsigned l = 0; l < nestLoops.size(); ++l) {
      bool inR = readLoopSet.count((int)l) != 0;
      bool inW = writeLoopSet.count((int)l) != 0;
      if (inR && !inW) reductionLoops.push_back((int)l);
      if (inW) outputLoops.push_back((int)l);
    }
    llvm::errs() << "[L1] detected: ";
    if (reductionLoops.empty()) {
      llvm::errs() << "stencil (no reduction dim; spatial: ";
      for (unsigned l = 0; l < nestLoops.size(); ++l)
        if (writeLoopSet.count((int)l) != 0) llvm::errs() << "L" << l << " ";
      llvm::errs() << ")";
    } else {
      llvm::errs() << "matmul (reduction dim: ";
      for (int l : reductionLoops) llvm::errs() << "L" << l << " ";
      llvm::errs() << ")";
    }
    llvm::errs() << "\n";

    // ---- 机器常数（一张卡校准一次，来源全部实测标注）----
    struct HW {
      double smBw = 7.21;         // GB/s — 单 SM 有效带宽（naive 实测 922.8/128）【机器类】
      double smCompute = 630.0;   // GFLOPS — 单 SM 计算吞吐【机器类】
      double rExposed = 330.0;    // ns — 基准延迟暴露（heat1d 十点拟合）【机器类，GEMM 待验】
      double aT = 0.60;           // ns/元素 — MLP 抵消系数（heat1d 拟合）【机器类，GEMM 待验】
      double pExp = 1.20;         // 深度摊销指数（heat1d 拟合）【机器类，GEMM 待验】
      double deltaFloor = 52.0;   // ns — 每 tile 不可隐藏成本下限（10-29 平台实测）【机器类】
      double launch = 3000.0;     // ns — Kernel 启动开销【机器类】
      int numSms = 128;
      int64_t smemCap = 99 * 1024;
    } hw;

    // ---- 顶层嵌套段划分工具（v1.7 提升到外层：组化分支与单 GEMM 分支共用）----
    // seg id = 环的最外层祖先在 nestLoops 中的下标；同段 = 同一顶层嵌套
    DenseMap<Operation *, int> loopOpIdx;
    for (unsigned i = 0; i < nestLoops.size(); ++i)
      loopOpIdx[nestLoops[i].getOperation()] = (int)i;
    auto outerIdxOf = [&](int l) -> int {
      int best = l;
      for (Operation *cur = nestLoops[l].getOperation()->getParentOp();
           cur != nullptr; cur = cur->getParentOp()) {
        auto it = loopOpIdx.find(cur);
        if (it != loopOpIdx.end()) best = it->second;  // 越上溯越外层 → 覆盖
      }
      return best;
    };

    // ============================================================
    // ===== A1 组化 pass-chain（v1.6，11-11）：按顶层嵌套段分组 =====
    // 结构事实（twomm 重推，11-10 速记纠正）：每 pass = 顶层嵌套
    //   (i,j,k)——out-out-red 两输出环 + 一归约环，另有 store-only
    //   清零段 → v1.1 交替假设 (nLoops==2P) 必拒（10 != 4）。
    // 设计定案（动码前三决策，11-11）：
    //   ① 分组 = 顶层嵌套段（每环上溯最外层祖先）：恰 1 归约环的段 =
    //      pass；0 归约环段 = 清零/辅助段，时间忽略（登记近似：512²
    //      纯写 MB 级，对候选排序为同额常数）；≥2 归约环段 → 交回旧拒因。
    //   ② 全链 GEMM 形（每 pass 恰 2 out 环）→ 本分支，每 pass 两态式
    //      与单 GEMM 模板逐位同源；全链 GEMV 形 / 混形 → 不置位
    //      groupedHandled，落回 v1.1 交替分支（atax 锚点由构造保证不变）。
    //   ③ 参数共享 v1：全 pass 共享 (BM,BN,TK,S)；每 pass 独立过 smem /
    //      半机守卫（任一不过 → 该配置不可行）；余数吸收照抄 B1。
    // 【smem 关键】同 memref 跨段多角色（twomm E：清零/pass1 输出/pass2
    //   流式面板）——全局合并 + last-wins 会把 E 判成 pass2 流式面板凭空
    //   翻倍（64×64/TK32/S4 下 164 KB > 99 KB 全灭）→ 必须按段独立建
    //   访问描述；段内窗口维含本段归约环 → ×(S+1)，否则 ×0
    //   （输出寄存器累加不占 smem，与单 GEMM 模板 11-04 修正同源）。
    // 【预注册】twomm：REJECT → Best 64×64/TK=32/S=4（每 pass 与
    //   gemm512 单 nest 逐位同构，排序必然一致），predicted ≈
    //   2×gemm512 行单 pass 预测 + 1×launch；七锚点逐位不变。
    // ============================================================
    bool groupedHandled = false;
    if (!reductionLoops.empty() && !windowFallback && reductionLoops.size() >= 2) {
      // 段划分（seg id = 段最外层环号，升序 = 程序序；loopOpIdx/outerIdxOf 已提升到外层）
      SmallVector<SmallVector<int, 4>, 8> segMembers(nestLoops.size());
      for (unsigned l = 0; l < nestLoops.size(); ++l)
        segMembers[outerIdxOf((int)l)].push_back((int)l);
      // 段分类 → pass 列表
      SmallVector<int, 4> passSegId;
      SmallVector<std::pair<int, int>, 4> passOutLoops;    // 每 pass (out0, out1)
      SmallVector<int, 4> passRedLoops;
      bool gemmChain = true, chainOk = true;
      for (unsigned seg = 0; seg < nestLoops.size() && chainOk && gemmChain; ++seg) {
        auto &members = segMembers[seg];
        if (members.empty()) continue;
        int redCnt = 0, redL = -1;
        SmallVector<int, 4> outs;
        for (int lm : members) {
          bool isRed = false;
          for (int r : reductionLoops)
            if (r == lm) { isRed = true; break; }
          if (isRed) { redCnt++; redL = lm; }
          else if (writeLoopSet.count(lm)) outs.push_back(lm);
        }
        if (redCnt == 0) continue;                         // ① 清零/辅助段
        if (redCnt > 1) { chainOk = false; break; }        // 嵌套多归约环 → 旧拒因
        if (outs.size() == 2) {                            // GEMM 形 pass
          passSegId.push_back((int)seg);
          passOutLoops.push_back({outs[0], outs[1]});
          passRedLoops.push_back(redL);
        } else {                                           // GEMV 形 / 异常 → 旧逻辑
          gemmChain = false; break;
        }
      }
      if (chainOk && gemmChain && (int)passSegId.size() >= 2) {
        int P = (int)passSegId.size();
        auto tripOfG = [&](int l) -> int64_t {
          auto t = nestLoops[l].getStaticTripCount();
          return t.has_value() ? (int64_t)t->getZExtValue() : -1;
        };
        bool constTrips = true;
        SmallVector<int64_t, 4> pM, pN, pK;
        for (int p = 0; p < P; ++p) {
          int64_t m = tripOfG(passOutLoops[p].first);
          int64_t n = tripOfG(passOutLoops[p].second);
          int64_t k = tripOfG(passRedLoops[p]);
          if (m <= 0 || n <= 0 || k <= 0) { constTrips = false; break; }
          pM.push_back(m); pN.push_back(n); pK.push_back(k);
        }
        if (!constTrips) {
          llvm::errs() << "\n[L1 Decision] REJECT (grouped pass-chain): 非常数 trip count\n";
          groupedHandled = true;
        } else {
          // ---- 按段重建访问描述（跨段合并会误伤双角色 memref，见上）----
          SmallVector<SmallVector<AccessDesc, 4>, 4> segAccesses;
          segAccesses.resize(P);
          DenseMap<int, int> segToPass;
          for (int p = 0; p < P; ++p) segToPass[passSegId[p]] = p;
          func->walk([&](Operation *op) {
            AffineMap amap;
            Value memref;
            SmallVector<Value, 4> mapOps;
            bool isStore = false;
            if (auto ld = dyn_cast<affine::AffineLoadOp>(op)) {
              amap = ld.getAffineMap(); memref = ld.getMemRef();
              for (Value v : ld.getMapOperands()) mapOps.push_back(v);
            } else if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
              amap = st.getAffineMap(); memref = st.getMemRef();
              for (Value v : st.getMapOperands()) mapOps.push_back(v);
              isStore = true;
            } else {
              return WalkResult::advance();
            }
            // 所属段：从 op 上溯第一个 AffineForOp（最内层）再取其段号
            int segLoop = -1;
            for (Operation *cur = op->getParentOp(); cur != nullptr;
                 cur = cur->getParentOp()) {
              auto it = loopOpIdx.find(cur);
              if (it != loopOpIdx.end()) { segLoop = it->second; break; }
            }
            if (segLoop < 0) return WalkResult::advance();
            int pIdx = -1;
            auto sp = segToPass.find(outerIdxOf(segLoop));
            if (sp != segToPass.end()) pIdx = sp->second;
            if (pIdx < 0) return WalkResult::advance();   // 清零/辅助段不进记账
            auto &accs = segAccesses[pIdx];
            size_t idx = accs.size();
            for (size_t a = 0; a < accs.size(); ++a)
              if (accs[a].memref == memref) { idx = a; break; }
            if (idx == accs.size()) {
              AccessDesc ad;
              ad.memref = memref;
              ad.elemBytes = cast<MemRefType>(memref.getType()).getElementTypeBitWidth() / 8;
              ad.dims.resize(amap.getNumResults());
              accs.push_back(std::move(ad));
            }
            AccessDesc &acc = accs[idx];
            if (isStore) acc.isWritten = true;
            else acc.isRead = true;
            for (unsigned d = 0; d < amap.getNumResults(); ++d) {
              int dimPos, c;
              if (!splitDimConst(amap.getResult(d), dimPos, c))
                return WalkResult::advance();  // 复杂表达式理论已被 windowFallback 拦截
              if (dimPos < 0) continue;
              Value iv = mapOps[dimPos];
              auto it = ivIndex.find(iv);
              if (it == ivIndex.end()) continue;
              acc.dims[d].push_back({it->second, c});
            }
            return WalkResult::advance();
          });
          // ---- 共享参数枚举：每 pass 两态式（与单 GEMM 模板逐位同源）----
          const int cBM[3] = {64, 128, 256};
          const int cBN[3] = {64, 128, 256};
          const int cTK[3] = {16, 32, 64};
          const double LminGemm = 1739.0;    // 与单 GEMM 模板同源（4096³ 斜率反解）
          double bestTime = 1e18;
          int bBM = 0, bBN = 0, bTK = 0, bS = 0;
          bool bestNonDiv = false;
          int nEnum = 0, nFeas = 0;   // D2（11-14）：枚举/可行计数 → Best 行双列口径
          for (int bi = 0; bi < 3; ++bi)
          for (int bj = 0; bj < 3; ++bj)
          for (int tk = 0; tk < 3; ++tk)
          for (int S = 1; S <= 4; ++S) {
            int BM = cBM[bi], BN = cBN[bj], TK = cTK[tk];
            if (BM == 256 && BN == 256) continue;  // 寄存器约束（同单模板）
            ++nEnum;
            double total = hw.launch * (double)P;
            bool feas = true;
            bool nonDiv = false;
            for (int p = 0; p < P && feas; ++p) {
              int64_t M = pM[p], N = pN[p], K = pK[p];
              SmallVector<int, 8> tiles(nestLoops.size(), 0);
              tiles[passRedLoops[p]] = TK;
              tiles[passOutLoops[p].first] = BM;
              tiles[passOutLoops[p].second] = BN;
              // smem：段内访问，窗口维含本段归约环 → ×(S+1)，否则 ×0
              int64_t smem = 0;
              double copyB = 0, outB = 0;
              for (auto &a : segAccesses[p]) {
                int64_t fp = windowFootprint(a, tiles);
                if (fp < 0) { feas = false; break; }
                bool str = false;
                for (size_t d = 0; d < a.dims.size() && !str; ++d)
                  for (auto &pr : a.dims[d])
                    if (pr.first == passRedLoops[p]) { str = true; break; }
                if (str) { smem += fp * (S + 1); copyB += (double)fp; }
                else outB += (double)fp;
              }
              if (!feas || smem > hw.smemCap) { feas = false; break; }
              int64_t numBlocks = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
              if (numBlocks < hw.numSms / 2) { feas = false; break; }  // 半机填充
              int64_t waves = (numBlocks + hw.numSms - 1) / hw.numSms;
              int64_t chunks = (K + TK - 1) / TK;                      // B1: ceil
              double copy = copyB / hw.smBw;
              double comp = 2.0 * (double)BM * (double)BN * (double)TK / hw.smCompute;
              double eff = std::min((double)S, std::max(1.0, (double)(chunks - 1)));
              double expo = std::max(hw.rExposed / std::pow(eff, hw.pExp), hw.deltaFloor);
              double cWrite = outB / hw.smBw;
              // 两态式（V0.3 ③，与单 GEMM 模板同源）：S=1 串行 / S≥2 重叠+Lcap
              double blockTime;
              if (S == 1)
                blockTime = (double)chunks * (copy + comp) + expo + cWrite;
              else {
                double Lcap = std::min(LminGemm, copy + comp);  // 重叠不劣于串行
                blockTime = (double)chunks * std::max({copy, comp, Lcap}) + expo + cWrite;
              }
              total += (double)waves * blockTime;
              if ((K % TK != 0) || (M % BM != 0) || (N % BN != 0)) nonDiv = true;
            }
            if (feas) ++nFeas;   // v1.8.2a：break 式守卫下只计真正可行的配置
            if (feas && total < bestTime) {
              bestTime = total; bBM = BM; bBN = BN; bTK = TK; bS = S;
              bestNonDiv = nonDiv;
            }
          }
          groupedHandled = true;
          if (bBM == 0) {
            llvm::errs() << "\n[L1 Decision] NO FEASIBLE CONFIG (grouped pass-chain)\n";
          } else {
            double flops = 0;
            for (int p = 0; p < P; ++p)
              flops += 2.0 * (double)pM[p] * (double)pN[p] * (double)pK[p];
            double tflops = flops / (bestTime * 1e-9) / 1e12;
            llvm::errs() << "\n[L1 Decision] Best: BM=" << bBM << " BN=" << bBN
                         << " TK=" << bTK << " Stages=" << bS << " predicted = "
                         << bestTime << " ns (" << tflops
                         << " TFLOPS, grouped pass-chain GEMM x" << P
                         << ") [搜索: 可行 " << nFeas << "/枚举 " << nEnum << "]\n";
            if (bestNonDiv)
              llvm::errs() << "[L1 Decision] 注意: 最优配置非整除（余数吸收）——"
                           << "发射需 separate 路径（待验证）\n";
          }
        }
      }
    }

    if (!reductionLoops.empty() && !windowFallback) {
      // ========================================================
      // ===== matmul 族模板分发（11-03 v1.1）=====
      //  单归约维 + 双输出循环 → GEMM 模板（原样 + B1 余数吸收）
      //  多归约维 + out/red 交替 pass 链（atax/mvt 类）→ GEMV 模板（新）
      //  其余结构 → 结构化拒因（不静默猜、不落 stencil）
      //  v1.6（11-11）：全 GEMM 形组化链已在上方分流（groupedHandled），
      //    GEMV 形链 / 混形 / 嵌套多归约环落回本分支，行为与 v1.5 一致
      //  v1.8（11-13）：GEMV 链改按段提取 out/red（顺序无关，bicg 可接）
      // ========================================================
      if (reductionLoops.size() >= 2 && !groupedHandled) {
        // ========================================================
        // ===== pass-chain GEMV 模板（v1.1→v1.8，atax/mvt/bicg 类）=====
        // 结构：P 个顺序 pass，每 pass = 1 输出环 + 1 归约环（顺序无关，
        //   v1.8 起按段提取，不再假设 out 外/red 内）；
        // 每 pass = 一次 GEMV（矩阵流式，输出 1D 向量）。
        // 决策：全 pass 共享 (T, TK, S)，总时间 = Σ pass；
        // smem 用窗口法（窗口维含归约环 → 流式 ×(S+1)，否则 ×1）。
        // 已知近似（如实登记）：1D memref 被多个兄弟环访问时，
        // 窗口取"最后归属环"的 tile（windowFootprint last-wins 行为）。
        // ========================================================
        int P = (int)reductionLoops.size();
        int nLoops = (int)nestLoops.size();
        bool altPattern = (nLoops == 2 * P);
        // v1.8（11-13）：按段提取 out/red（替代奇偶位假设）——每段恰 1 red + 1 out
        //   即 GEMV 形，顺序无关（bicg pass1 红外/写内也能接）。
        //   兼容性：atax/mvt 的 out 恰在偶位、red 在奇位 → 提取结果与旧逻辑
        //   逐位相同（tiles/流式判定/passM/K 全同）→ 锚点构造级安全。
        //   仍拒：GEMM 形段（2 out，组化分支处理）、辅助清零段（0 red，
        //   未来与组化分支行为对齐时再容忍）。
        SmallVector<int, 8> gemvOut, gemvRed;   // 每 pass 的 out/red 环（按段序）
        if (altPattern) {
          SmallVector<SmallVector<int, 4>, 8> segMembers(nLoops);
          for (int l = 0; l < nLoops; ++l)
            segMembers[outerIdxOf(l)].push_back(l);
          int pCnt = 0;
          for (int seg = 0; seg < nLoops && altPattern; ++seg) {
            auto &members = segMembers[seg];
            if (members.empty()) continue;
            int redCnt = 0, redL = -1, outCnt = 0, outL = -1;
            for (int lm : members) {
              bool isRed = false;
              for (int r : reductionLoops)
                if (r == lm) { isRed = true; break; }
              if (isRed) { redCnt++; redL = lm; }
              else if (writeLoopSet.count(lm)) { outCnt++; outL = lm; }
            }
            if (redCnt == 1 && outCnt == 1) {
              gemvOut.push_back(outL); gemvRed.push_back(redL); pCnt++;
            } else {
              altPattern = false;   // GEMM 形段 / 辅助段 / 异常段 → 旧拒因
            }
          }
          if (pCnt != P) altPattern = false;
        }
        if (!altPattern) {
          llvm::errs() << "\n[L1 Decision] REJECT (matmul-family): " << P
                       << " 个归约维且非 out/red 交替 pass 链（v1.1 不支持）\n";
        } else {
          auto tripOfG = [&](int l) -> int64_t {
            auto t = nestLoops[l].getStaticTripCount();
            return t.has_value() ? (int64_t)t->getZExtValue() : -1;
          };
          SmallVector<int64_t, 8> passM, passK;
          bool constTrips = true;
          for (int p = 0; p < P; ++p) {
            int64_t m = tripOfG(gemvOut[p]), k = tripOfG(gemvRed[p]);
            if (m <= 0 || k <= 0) { constTrips = false; break; }
            passM.push_back(m); passK.push_back(k);
          }
          if (!constTrips) {
            llvm::errs() << "\n[L1 Decision] REJECT (pass-chain): 非常数 trip count\n";
          } else {
            // GEMV 候选：T 放宽到 8（GEMV 并行度天然低，小 T 合法）。
            // 降序枚举（v1.4 支配 tie-break）：模型同分的配置中，段宽更大者
            // 在被省略的二阶效应（DRAM 段宽效率）上单调占优——保留宽段配置。
            const int cT[6] = {256, 128, 64, 32, 16, 8};
            const int cTK[3] = {64, 32, 16};
            double bestTime = 1e18;
            int bT = 0, bTK = 0, bS = 0;
            int nEnum = 0, nFeas = 0;   // D2（11-14）：枚举/可行计数 → Best 行双列口径
            for (int ti = 0; ti < 6; ++ti)
            for (int tki = 0; tki < 3; ++tki)
            for (int S = 1; S <= 4; ++S) {
              int T = cT[ti], TK = cTK[tki];
              ++nEnum;
              // smem：v1.8.5（11-15）——per-launch 口径，与时间公式 P×launch 的独立
              // 发射结构一致：单 launch = A 面板 T×TK（列向 pass 为 TK×T，等元素数）
              // + 向量 chunk TK，流式 ×(S+1)。
              // 【审计 11-15】旧口径（全 pass 合并窗口求和 + 转置读 max-tile 膨胀，
              //   A 窗口 max(T,TK)²）保守截断可行集（atax 48/72 vs 载具实测 57/72）。
              //   实测裁决（atax_sweep 双日志）：T=128/TK=64 实测 779–820 µs 比
              //   T=64 慢 ~40%（numBlocks=64 半机压线 → occ=0.5），模型 occ 惩罚
              //   预测 1179 µs → 决策不翻转（T=128 劣势由 occ 覆盖，非 smem），
              //   口径对齐安全；段宽经验式的独立验证归 A6。
              int64_t smem = (int64_t)(S + 1) * ((int64_t)T * TK + TK) * 4;
              if (smem > hw.smemCap) continue;
              // 每 pass：守卫 + GEMV 两资源模型（V0.3 ⑤，11-10 深夜），求和
              // 【机理】11-10 atax N=8192 穷举（57 配置实测）定案：
              //   pass_time = max(chunks × L_min, stream_bytes / BW_eff) + wback
              //   L_min ≈ 820 ns = chunk 流水最小周期（wait_prior + sync×2 +
              //     cp.async 发射 + 流水泡），与 chunk 大小无关——实测 TK=16 列
              //     per-chunk 恒 ~820 ns 跨 T 不变；T=8 薄流水修正 1260 ns。
              //   BW_eff = BW_agg × min(1, numBlocks/numSms) × max(0.6, 1−2.4/T)
              //     （DRAM 聚合带宽 × 占据率 × 段宽效应经验式——后者三点拟合
              //     T=16/32/64，【待精化】pass1/pass2 段宽分离曲线留 V0.3）。
              //   两资源取 max：SM 流水下界 vs DRAM 上界，先到者为瓶颈。
              // 实测验证：T≥16 的 12 个 (T,TK) 组合 6% 内；解释全部结构
              // （TK=16→64 提速 33%、T≥32 不变性、S 不敏感性）。
              // S 维：实测 GEMV 全域不敏感（展宽 ≤3%）→ 并列取 1（枚举序保留）。
              double total = hw.launch * P;
              bool guardOk = true;
              for (int p = 0; p < P && guardOk; ++p) {
                int64_t M = passM[p], K = passK[p];
                int64_t numBlocks = (M + T - 1) / T;
                if (numBlocks < hw.numSms / 2) { guardOk = false; break; }
                int64_t chunks = (K + TK - 1) / TK;
                double Lmin = (T < 16) ? 1260.0 : 820.0;
                double occ = std::min(1.0, (double)numBlocks / (double)hw.numSms);
                double bwEff = (double)hw.numSms * hw.smBw * occ
                             * std::max(0.6, 1.0 - 2.4 / (double)T);
                double streamB = (double)M * (double)K * 4.0;   // A 全量（向量项并入常数）
                double passTime = std::max((double)chunks * Lmin, streamB / bwEff)
                                  + (double)T * 4.0 / hw.smBw;  // 输出写回
                total += passTime;
              }
              if (guardOk) ++nFeas;   // v1.8.2a：break 式守卫下只计真正可行的配置
              if (guardOk && total < bestTime) {
                bestTime = total; bT = T; bTK = TK; bS = S;
              }
            }
            if (bT == 0) {
              // 结构化拒因：区分并行度不足 vs smem 超限
              int64_t maxBlocks = 0;
              for (int64_t m : passM) {
                int64_t nb = (m + 7) / 8;   // 最小 T=8 下的 block 数上限
                if (nb > maxBlocks) maxBlocks = nb;
              }
              llvm::errs() << "\n[L1 Decision] REJECT (pass-chain): 无可行配置——"
                           << "并行度不足（最小 T=8 时 numBlocks=" << maxBlocks
                           << " < " << hw.numSms / 2 << "）或 smem 超限\n";
            } else {
              // 11-12 修复：flops 逐 pass 求和——原 passM×passK 叉积 = P² 组合，
              //   P=2 时 TFLOPS 虚高 2×（atax_big 8.794383e-01 → ~4.40e-01，
              //   mvt_probe 8.013917e-02 → ~4.01e-02，均为预期内的一行显示差异；
              //   时间模型与决策不受影响）。纯显示修复，其余输出逐位不变。
              double flops = 0;
              for (int p = 0; p < P; ++p)
                flops += 2.0 * (double)passM[p] * (double)passK[p];
              double tflops = flops / (bestTime * 1e-9) / 1e12;
              llvm::errs() << "\n[L1 Decision] Best: T=" << bT << " TK=" << bTK
                           << " Stages=" << bS << " predicted = " << bestTime
                           << " ns (" << tflops << " TFLOPS, pass-chain GEMV x"
                           << P << ") [搜索: 可行 " << nFeas << "/枚举 " << nEnum << "]\n";
            }
          }
        }
      }
      if (reductionLoops.size() == 1) {
      // ========================================================
      // ===== matmul 模板：4D 决策（公式照抄 l2_gemm.py，K-chunk 粒度）=====
      // 11-03 B1：整除约束移除（余数吸收），chunks 改 ceil；
      //      smem 窗口法 dimCap 封顶天然保守；发射端 separate 路径待验证
      // ========================================================
      int redLoop = reductionLoops[0];
      // v1.7（11-12）：outLoops = 与归约环同顶层嵌套段且被写的环——
      //   修复独立清零段（store-only 顶层嵌套）被计入 outLoops 的盲区
      //   （symm_true v2 实证：清零段致 outLoops=4 → "output loops != 2" 拒因）。
      //   单嵌套 kernel（gemm4096/512/500/1024）同段过滤后集合不变 → 九发锚点逐位安全。
      //   登记近似：守卫形式（常 trip + affine.if）的守卫浪费不计入模型，预测偏乐观。
      int redSeg = outerIdxOf(redLoop);
      SmallVector<int, 4> outLoops;
      for (unsigned l = 0; l < nestLoops.size(); ++l)
        if ((int)l != redLoop && outerIdxOf((int)l) == redSeg &&
            writeLoopSet.count((int)l))
          outLoops.push_back((int)l);

      if (outLoops.size() != 2) {
        llvm::errs() << "[L1 Decision] matmul: output loops != 2 (v1 不支持)\n";
      } else {
        auto tripOf = [&](int l) -> int64_t {
          auto t = nestLoops[l].getStaticTripCount();
          return t.has_value() ? (int64_t)t->getZExtValue() : -1;
        };
        int64_t redTrip = tripOf(redLoop);
        int64_t trip0 = tripOf(outLoops[0]);
        int64_t trip1 = tripOf(outLoops[1]);

        if (redTrip <= 0 || trip0 <= 0 || trip1 <= 0) {
          llvm::errs() << "[L1 Decision] matmul: non-constant trip counts (v1 不支持)\n";
        } else {
          const int cBM[3] = {64, 128, 256};
          const int cBN[3] = {64, 128, 256};
          const int cTK[3] = {16, 32, 64};
            double bestTime = 1e18;
            int bBM = 0, bBN = 0, bTK = 0, bS = 0;
            bool bestNonDiv = false;
            int nEnum = 0, nFeas = 0;   // D2（11-14）：枚举/可行计数 → Best 行双列口径
            // V0.3 ⑤-GEMM（11-10 深夜）：chunk 流水下界 L_min 移植——修复 TK 残余。
            // 【机理】同 GEMV 两资源（atax 57 点实测定案）：每 chunk 固定流水开销
            //   （wait + sync×2 + cp.async 发射 + 泡）→ 小 TK 的多 chunk 劣化。
            //   GEMM 实测斜率 4096³ 64×64 S=2: TK=16→64 = −11%（6098→5414 µs）。
            // 【标定】L_min(GEMM) = 1739 ns——4096³ 64×64 S=2 斜率反解，可行带
            //   [1188, 2205]；决策对 L ∈ [1500, 2270] 鲁棒（512/500 均落 (64,32,4)）。
            //   与 GEMV 的 820 ns 异值：kernel 结构不同（双面板 cp.async + 寄存器
            //   分块累加 vs 单向量累加），各自标定，登记。
            // 【形式】仅重叠态加下界，且下界不超过串行和（重叠不劣于串行的物理约束）：
            //   S≥2: chunk = max(copy, comp, min(L_min, copy + comp))
            // 【预注册】512/500 → 决策 (64,32,4)：实测 25.4 / 29.7 µs = rank 3/48 =
            //   6.3%、距优 +5.4% / +9.6% → 双达标（原 rank 8/16.7%、7/+24%）。
            //   4096：copy(3408) > L 不绑定，决策不变 (128,256,16,3)。
            const double LminGemm = 1739.0;

            // v1.8.3（11-15）：双面板记账——同一 memref 被不同"逐维载体集合"访问时，
            // 各需独立 smem 面板（syrk: A[i,k] + A[j,k] → (BM+BN)×TK，而非合并的
            // max(BM,BN)×TK）。11-15 stril oracle 实证：合并窗口低估 smem——
            // (64,64,TK=64,S=4) 真实 160KB > 99KB 被载具正确跳过而模型报可行
            // （"误判可行"威胁零非法招牌，比低估更危险）。分组键 = (memref,
            // 逐维载体环集合)；组内窗口并集（D4 max-tile 语义）。作用域：仅本分支
            // （组化分支 twomm/3mm 锚点、GEMV 分支 atax/mvt/bicg 锚点不经过此处）。
            // 面板流量口径同步按组求和 → syrk A 双面板 = 流两次（真实语义）。
            SmallVector<AccessDesc, 8> panelGroups;
            for (auto &p : winPanels) {
              int g = -1;
              for (unsigned gi = 0; gi < panelGroups.size() && g < 0; ++gi) {
                if (panelGroups[gi].memref != p.memref) continue;
                bool same = panelGroups[gi].dims.size() == p.dims.size();
                for (size_t d = 0; same && d < p.dims.size(); ++d) {
                  int pc = p.dims[d].empty() ? -1 : p.dims[d][0].first;  // per-op 每维至多 1 载体
                  if (pc < 0) { same = panelGroups[gi].dims[d].empty(); continue; }
                  bool found = false;
                  for (auto &pr : panelGroups[gi].dims[d])
                    if (pr.first == pc) { found = true; break; }
                  if (!found) same = false;
                }
                if (same) g = (int)gi;
              }
              if (g < 0) {
                panelGroups.push_back(p);
              } else {
                for (size_t d = 0; d < p.dims.size(); ++d)
                  if (!p.dims[d].empty())
                    panelGroups[(size_t)g].dims[d].push_back(p.dims[d][0]);  // 组内并集
              }
            }

          for (int bi = 0; bi < 3; ++bi)
          for (int bj = 0; bj < 3; ++bj)
          for (int tk = 0; tk < 3; ++tk)
          for (int S = 1; S <= 4; ++S) {
            int BM = cBM[bi], BN = cBN[bj], TK = cTK[tk];
            // 寄存器约束（10-31 实证：256×256 → 每线程 256 累加寄存器 > 255 硬上限，
            // 编译失败/溢出——Python 参考模型同款排除，V0.3 ② 落地）
            if (BM == 256 && BN == 256) continue;
            ++nEnum;
            // 11-03 B1：整除约束移除——非整除时 ceil 分块 + 边界 tile 余数吸收；
            // smem 窗口法 dimCap 封顶 = 天然保守（部分 tile 只会更小）

            SmallVector<int, 4> tiles(nestLoops.size(), 0);
            tiles[redLoop] = TK;
            tiles[outLoops[0]] = BM;
            tiles[outLoops[1]] = BN;

            // smem：流式面板（窗口维含归约循环）×(S+1)，按 v1.8.3 面板组求和；
            // 输出 C 寄存器累加不占 smem（×0，11-04 修正——与 sgemm_tiled 实际分配逐字节对齐：
            // kernel 打印 16/24/48/72/96 KB 与 streams×(S+1) 精确一致）；C 写回计时间公式的 DRAM 流量
            int64_t smem = 0;
            bool feas = true;
            SmallVector<int64_t, 8> fps;
            SmallVector<bool, 8> isStream;
            for (auto &a : panelGroups) {
              int64_t fp = windowFootprint(a, tiles);
              if (fp < 0) { feas = false; break; }
              bool str = false;
              for (size_t d = 0; d < a.dims.size() && !str; ++d)
                for (auto &pr : a.dims[d])
                  if (pr.first == redLoop) { str = true; break; }
              fps.push_back(fp);
              isStream.push_back(str);
              smem += fp * (str ? (S + 1) : 0);
            }
            if (!feas || smem > hw.smemCap) continue;

            int64_t numBlocks = ((trip0 + BM - 1) / BM) * ((trip1 + BN - 1) / BN);
            if (numBlocks < hw.numSms / 2) continue;          // 半机填充（P4 语义）
            ++nFeas;

            int64_t waves = (numBlocks + hw.numSms - 1) / hw.numSms;
            int64_t chunks = (redTrip + TK - 1) / TK;         // 11-03 B1: ceil

            double copyB = 0, outB = 0;
            for (size_t a = 0; a < panelGroups.size(); ++a)
              (isStream[a] ? copyB : outB) += (double)fps[a];
            double copy   = copyB / hw.smBw;
            double comp   = 2.0 * (double)BM * (double)BN * (double)TK / hw.smCompute;
            double eff    = std::min((double)S, std::max(1.0, (double)(chunks - 1)));
            double expo   = std::max(hw.rExposed / std::pow(eff, hw.pExp), hw.deltaFloor);
            double cWrite = outB / hw.smBw;
            // V0.3 ③（11-10）：两态 GEMM 公式——修复"决策恒 S=1"的 max 形式盲区。
            // 机理（11-09 诊断，1108 文档第十一节）：旧式 chunks×max(copy,comp,expo)
            // 在任意 S 下都假设搬运/计算完全重叠，而该重叠恰是深度 S 买来的
            // （S=1 逐 chunk 串行等搬运；实测 4096³ 64×64 组 S=1→2 = −17.5%）。
            // 且旧式 expo 恒被 copy 隐藏（copy/comp ≥ 1.37、expo ≤ 330 < copy）
            // → S 维零分辨率。两态：
            //   S=1（串行）: block = chunks×(copy + comp) + expo + cWrite
            //   S≥2（重叠）: block = chunks×max(copy, comp) + expo + cWrite
            // expo 移出 max 单列，eff(S) 摊销保留。
            // 标定/验证：4096³ 全扫参拟合；512³/500³ 留出验证（存量 sweep 日志，
            // 零新 GPU 实验）。
            double blockTime;
            if (S == 1)
              blockTime = (double)chunks * (copy + comp) + expo + cWrite;
            else {
              double Lcap = std::min(LminGemm, copy + comp);   // 重叠不劣于串行
              blockTime = (double)chunks * std::max({copy, comp, Lcap}) + expo + cWrite;
            }
            double total = hw.launch + (double)waves * blockTime;

            if (total < bestTime) {
              bestTime = total; bBM = BM; bBN = BN; bTK = TK; bS = S;
              bestNonDiv = (redTrip % TK != 0) || (trip0 % BM != 0) || (trip1 % BN != 0);
            }
          }

          if (bBM == 0) {
            llvm::errs() << "\n[L1 Decision] NO FEASIBLE CONFIG (matmul template)\n";
          } else {
            double tflops = 2.0 * (double)redTrip * (double)trip0 * (double)trip1
                            / (bestTime * 1e-9) / 1e12;
            llvm::errs() << "\n[L1 Decision] Best: BM=" << bBM << " BN=" << bBN << " TK=" << bTK
                         << " Stages=" << bS << " predicted = " << bestTime << " ns ("
                         << tflops << " TFLOPS) [搜索: 可行 " << nFeas
                         << "/枚举 " << nEnum << "]\n";
            if (bestNonDiv)
              llvm::errs() << "[L1 Decision] 注意: 最优配置非整除（余数吸收）——"
                           << "发射需 separate 路径（待验证，见 run_pipeline v3 注释）\n";
          }
        }
      }
      } // reductionLoops.size() == 1 (GEMM 模板)
    } else {
      // ========================================================
      // ===== stencil 族分发（v1.2）：按空间环数分流 =====
      //   空间环 = IV 出现在写下标的循环（stencil 的写经过空间环）
      //   1 个空间环 → 1D 模板（原样，回归锚点）
      //   2 个空间环 + 时间环 L0 → stencil-2d 模板（新，jacobi-2d 类）
      //   ≥3 → 结构化拒因（3D 待 v1.3）
      // ========================================================
      SmallVector<int, 4> spatialLoops;
      for (unsigned l = 0; l < nestLoops.size(); ++l)
        if (writeLoopSet.count((int)l) != 0) spatialLoops.push_back((int)l);

      // 嵌套 2D 特征（v1.2.1）：每个 memref 维至多由一个循环携带。兄弟环结构
      // （如 jacobi-1d 的交换双缓冲：t 内两个 i 环先后写 B/A）会把同一维交给
      // 多个环先后携带——那是 pass-chain stencil，不是 2D。
      // 【教训入库 11-08】空间环计数不是 2D 的充分条件，jacobi1d 曾被误判 2D。
      bool dimCarried2d = true;
      for (auto &a : winAccesses)
        for (size_t d = 0; d < a.dims.size(); ++d) {
          int carrier = -1;
          for (auto &pr : a.dims[d]) {
            if (carrier == -1) carrier = pr.first;
            else if (pr.first != carrier) dimCarried2d = false;
          }
        }
      bool siblingSwap = (spatialLoops.size() == 2 && !dimCarried2d);
      if (siblingSwap)
        llvm::errs() << "\n[L1] stencil 兄弟环结构（交换双缓冲类）→ 兄弟交换 1D 模板"
                     << "（v1.3 窗口化记账；多 pass 时间步 launch 语义仍待登记）\n";

      if (spatialLoops.size() >= 3 && !siblingSwap) {
        llvm::errs() << "\n[L1 Decision] REJECT (stencil-3d): " << spatialLoops.size()
                     << " 个空间维（v1.2 仅支持 1D/2D）\n";
      } else if (spatialLoops.size() == 2 && dimCarried2d) {
      // ========================================================
      // ===== stencil-2d 模板（v1.2，jacobi-2d 类）=====
      // 结构要求：L0 = 时间环（IV 不进任何下标，trip = 时间步数），
      //           L1/L2 = 空间环（trip = 空间尺寸）。
      // 候选：方形 tile Ti=Tj ∈ {16,32,64,128} × S ∈ 1..4（16 组合）。
      // 公式 = heat1d 加法形式推广：per-tile = copy + comp + expo，
      //        R = max(330 − 0.6×T, 1) 的 T_eff 取流水环长度——
      //        【待验点】2D 暴露延迟的 tile 依赖未经实测，jacobi-2d 首轮实验检验。
      // smem：窗口含空间环的 memref = 流式 ×(S+1)——stencil 的输出数组下标
      //        含空间环，按流式算物理正确（C-in-smem 教训的正面场景）；
      //        已知保守偏差：相邻 tile 的 halo 窗口重叠 → 流量小幅高估（方向安全）。
      // ========================================================
      auto tripOf2d = [&](int l) -> int64_t {
        auto t = nestLoops[l].getStaticTripCount();
        return t.has_value() ? (int64_t)t->getZExtValue() : -1;
      };
      bool shape2d = (nestLoops.size() == 3 && spatialLoops[0] == 1 && spatialLoops[1] == 2
                      && !readLoopSet.count(0) && !writeLoopSet.count(0));
      int64_t tSteps = tripOf2d(0), dim1 = tripOf2d(1), dim2 = tripOf2d(2);
      if (!shape2d) {
        llvm::errs() << "\n[L1 Decision] REJECT (stencil-2d): 结构不符"
                     << "（需要 时间环 L0 + 空间环 L1 L2 的三层嵌套）\n";
      } else if (tSteps <= 0 || dim1 <= 0 || dim2 <= 0) {
        llvm::errs() << "\n[L1 Decision] REJECT (stencil-2d): 非常数 trip count\n";
      } else {
        const int cT2[4] = {16, 32, 64, 128};
        double bestTime = 1e18;
        int bT = 0, bS = 0;
        for (int ti = 0; ti < 4; ++ti)
        for (int S = 1; S <= 4; ++S) {
          int T = cT2[ti];
          SmallVector<int, 4> tiles(nestLoops.size(), 0);
          tiles[1] = T; tiles[2] = T;
          int64_t smem = 0, copyB = 0;
          bool feas = true;
          for (auto &a : winAccesses) {
            int64_t fp = windowFootprint(a, tiles);
            if (fp < 0) { feas = false; break; }
            copyB += fp;                   // 读写都计搬运流量（A 窗口读 + B tile 写）
            // v1.2.2 smem 修正：只计被读（staged）的窗口 ×(S+1)。
            // 【11-08 修正】v1.2 把只写的 B 也计 ×(S+1) 属 C-in-smem 同型反向
            // 错误——stencil tile 结果直接写回全局（块内零复用），B 不进 smem；
            // 虚高 smem 曾把深度 S 系统性误拒（模型因此错判 S=1）。
            if (a.isRead) smem += fp * (S + 1);
          }
          if (!feas || smem > hw.smemCap) continue;
          int64_t numBlocks = ((dim1 + T - 1) / T) * ((dim2 + T - 1) / T);
          if (numBlocks < hw.numSms / 2) continue;             // 半机填充守卫（P4 语义）
          int64_t waves = (numBlocks + hw.numSms - 1) / hw.numSms;
          double copy = (double)copyB / hw.smBw;
          double comp = flops_per_iter * (double)T * (double)T / hw.smCompute;
          double eff  = std::min((double)S, std::max(1.0, (double)(waves - 1)));
          double R    = std::max(hw.rExposed - hw.aT * (double)T, 1.0);   // T_eff = 流水环长度【待验点】
          double expo = std::max(R / std::pow(eff, hw.pExp), hw.deltaFloor);
          double total = hw.launch + (double)tSteps * waves * (copy + comp + expo);
          if (total < bestTime) { bestTime = total; bT = T; bS = S; }
        }
        if (bT == 0) {
          llvm::errs() << "\n[L1 Decision] NO FEASIBLE CONFIG (stencil-2d template)\n";
        } else {
          llvm::errs() << "\n[L1 Decision] Best: Tile2d=" << bT << "," << bT
                       << " Stages=" << bS << " predicted = " << bestTime
                       << " ns (stencil-2d)\n";
        }
      }
      } else if (spatialLoops.size() == 2 && siblingSwap) {
      // ========================================================
      // ===== 兄弟交换 1D 模板（v1.3，11-09：窗口化记账）=====
      // 结构：时间环 L0（IV 不进任何下标，trip = 时间步数）+ 两个兄弟空间环
      //       L1/L2（交换双缓冲：pass1 读 A 写 B，pass2 读 B 写 A）
      //       ——真 PolyBench jacobi-1d 形态。
      // 【11-09 修复】原回落 legacy 旧记账（base = 全数组）→ 大 N 全灭
      //   （N=65536 诊断：base=262KB → smem 需求 2MB → 24 候选全拒）。
      //   改窗口法（stencil-2d 同款）：每 memref 窗口 = (T+2)×4（halo ±1，
      //   两个 memref 跨 pass 都被读，per-memref 粒度按都读保守计 smem）。
      // 已登记假设/局限：
      //   a) FLOPs/元素/pass 由 flops_per_iter 反推（×interior/2），假设
      //      = 2 兄弟 pass、两者 trip 相等、每 pass 每元素算术量相同；
      //   b) 时间步 launch 语义：total 只计一次 hw.launch（真实执行
      //      2×tSteps 次发射）——legacy 缺陷不在此修（V0.3 候选⑥），
      //      小尺寸多步下模型失真（11-08 实测 33×）；
      //   c) 无半机填充守卫（小 N 占用率低时模型精度受限——如实登记，
      //      决策质量验证在 N ≥ 2^24 进行，见 1108 文档第十节）；
      //   d) 相邻 tile halo 窗口重叠 → 流量保守高估（方向安全，同 stencil-2d）；
      //   e) 写侧流量按窗口计（含 halo），~4% 保守高估（方向安全）。
      // 输出格式保持 "Best: Tile=... Stages=..."（glue stencil 分支零改动）。
      // ========================================================
      if (nestLoops.size() != 3 || spatialLoops[0] != 1 || spatialLoops[1] != 2) {
        llvm::errs() << "\n[L1 Decision] REJECT (stencil-sibling): 结构不符"
                     << "（需要 时间环 L0 + 两个兄弟空间环 L1 L2）\n";
      } else {
        auto tripOfS = [&](int l) -> int64_t {
          auto t = nestLoops[l].getStaticTripCount();
          return t.has_value() ? (int64_t)t->getZExtValue() : -1;
        };
        int64_t tSteps = tripOfS(0);
        int64_t interior = tripOfS(spatialLoops[0]);
        if (tSteps <= 0 || interior <= 0) {
          llvm::errs() << "\n[L1 Decision] REJECT (stencil-sibling): 非常数 trip count\n";
        } else {
          // FLOPs/元素/pass：totalFLOPs = flops_per_iter × tSteps × interior²，
          //   每 timestep 2 pass × interior 元素 → flopPerElemPass = flops_per_iter × interior / 2
          //   （假设 a：jacobi-1d 验证 = 3.0，即 2 addf + 1 mulf ✓）
          double flopPerElemPass = flops_per_iter * (double)interior / 2.0;
          double bestTime = 1e18;
          int bestT = 0, bestS = 0;
          for (int S = 1; S <= 4; ++S) {
            for (int T = 16; T <= (int)interior + 1; T *= 2) {
              SmallVector<int, 4> tiles(nestLoops.size(), 0);
              tiles[spatialLoops[0]] = T; tiles[spatialLoops[1]] = T;
              int64_t smem = 0, fpSum = 0;
              bool feas = true;
              for (auto &a : winAccesses) {
                int64_t fp = windowFootprint(a, tiles);
                if (fp < 0) { feas = false; break; }
                fpSum += fp;
                if (a.isRead) smem += fp * (S + 1);   // 只计被读窗口（v1.2.2 同款）
              }
              if (!feas || smem > hw.smemCap) continue;
              // 流量：每 timestep 2 pass，每 pass = 读一窗口 + 写一 tile；
              // per-memref fp 已含 halo，按 2×Σfp 保守计（假设 e）
              int64_t copyStepB = 2 * fpSum;
              int64_t numTiles = (interior + T - 1) / T;
              int64_t waves = (numTiles + hw.numSms - 1) / hw.numSms;
              double eff  = std::min((double)S, std::max(1.0, (double)(waves - 1)));
              double copy = (double)copyStepB / hw.smBw;
              double comp = 2.0 * flopPerElemPass * (double)T / hw.smCompute;
              double R    = std::max(hw.rExposed - hw.aT * (double)T, 1.0);
              double expo = std::max(R / std::pow(eff, hw.pExp), hw.deltaFloor);
              double total = hw.launch + (double)tSteps * (double)waves * (copy + comp + expo);
              if (total < bestTime) { bestTime = total; bestT = T; bestS = S; }
            }
          }
          if (bestT == 0) {
            llvm::errs() << "\n[L1 Decision] NO FEASIBLE CONFIG (stencil-sibling)\n";
          } else {
            llvm::errs() << "\n[L1 Decision] Best: Tile=" << bestT << " Stages=" << bestS
                         << " predicted = " << bestTime
                         << " ns (stencil-sibling, window)\n";
          }
        }
      }
      } else {
      // ========================================================
      // ===== stencil 模板：1D 决策（原样保留，回归锚点）=====
      // ========================================================
      double bestTime = 1e18;
      int bestT = 0, bestS = 0;

      for (int S = 1; S <= 4; S++) {
        for (int T = 16; T <= N; T *= 2) {
          int64_t bytes = 0;
          for (auto &[base, elem] : memrefs)
            bytes += base + (int64_t)(T - 1) * elem;

          int64_t smem = bytes * (S + 1);
          if (smem > hw.smemCap) continue;

          int64_t numTiles = (N + T - 1) / T;
          int waves = (numTiles + hw.numSms - 1) / hw.numSms;
          if (N >= 8192 && numTiles < 512) continue;

          double eff  = std::min((double)S, std::max(1.0, (double)(waves - 1)));
          double copy = (double)bytes / hw.smBw;
          double comp = (double)flops_per_iter * T / hw.smCompute;
          // V0.2 校准形式（加法）：tile = 搬运 + 计算 + 暴露延迟
          double R    = std::max(hw.rExposed - hw.aT * (double)T, 1.0);
          double expo = std::max(R / std::pow(eff, hw.pExp), hw.deltaFloor);
          double total = hw.launch + waves * (copy + comp + expo);

          if (total < bestTime) {
            bestTime = total;
            bestT = T;
            bestS = S;
          }
        }
      }

      if (bestT == 0) {
        llvm::errs() << "\n[L1 Decision] NO FEASIBLE CONFIG: all candidates rejected"
                     << " (smem/capacity checks)\n";
      } else {
        llvm::errs() << "\n[L1 Decision] Best: Tile=" << bestT << " Stages=" << bestS
                     << " predicted = " << bestTime << " ns\n";
      }
      } // stencil 1D
    } // stencil 族分发
  }
};
} // namespace

static PassRegistration<L1FeasPass> reg;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<affine::AffineDialect, func::FuncDialect, memref::MemRefDialect, arith::ArithDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "mlir-l1feas", registry));
}
