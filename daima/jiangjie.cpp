W      = (M·K + K·N + M·N)·4                     工作集
bw     = smBw × (L2_BOOST if W ≤ L2_BYTES else 1)  有效每 SM 带宽
nb     = ceil(M/BM) · ceil(N/BN)                  块数
waves  = ceil(nb / numSms)                        波次
chunks = ceil(K / TK)                             每 block 切几段

copy   = Σ(流式面板字节) / bw + cChunk            每 chunk 搬运
comp   = 2·BM·BN·TK / smCompute                   每 chunk 计算
eff    = min(S, max(1, chunks−1))                 实际可重叠的深度
expo   = max(rExposed / eff^pExp, deltaFloor)      每 block 暴露延迟
cWrite = Σ(输出面板字节) / smBw                    写回（不加 L2 boost）

S = 1 ：block = chunks·(copy + comp) + expo + cWrite
S ≥ 2 ：block = chunks·max(copy, comp, min(LminGemm, copy+comp)) + expo + cWrite