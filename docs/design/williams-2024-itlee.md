# Williams 2024 ITLEE 几何容差

| 项目 | 内容 |
| --- | --- |
| 状态 | 实现中 |
| 分支 | `feat/williams-2024-itlee`（基于 `origin/main`） |
| 目标模块 | `graphatoms.geometry`、`graphatoms.system`、`graphatoms.enterpoint.network` |
| 参考 | Williams, C. J. *Off-lattice kinetic Monte Carlo methods and the Fe-H system*. PhD thesis, University of Cambridge, 2024, §3.1.2, pp. 33–47 |

## 1. 目标

把 GraphAtoms 的事件复用从“精确图哈希判等”扩展为：

```text
L0 图哈希 / 图同构      → 候选与精确身份
L1 LAD/VF2 子图同构      → pattern → system 的原子对应 π_k
L2 ITLEE + Procrustes    → 几何容差校验，保留 l2 ≤ δ 的映射
L3 rmsd/重建守卫         → 失败则记录并折半该 LE 的 δ
```

`SysGraph.is_equivalent` 提供两种等大对象判等：

- `method="graph"`：图同构；`hash` 只做快速否决，不是判等；
- `method="itlee"`：带颜色点云的 ITLEE 几何等价。

不等大 pattern 仍由 `get_match_mode` 处理：LAD 给对应关系，ITLEE 只做 L2 校验，不做拓扑无关的全系统搜索。

## 2. ITLEE 数学定义

存在正交矩阵 `O` 与置换 `π`，使

```text
Σ_i ||p_i - O q_π(i)||² ≤ δ²
```

实现使用 `delta` 作为整个残差向量的 L2 范数，返回

```text
rmsd = l2 / sqrt(n)
```

内距必要条件（Williams 式 3.12）：

```text
|p_ij - q_ij| ≤ sqrt(2) * δ
```

δ 的物理上界（式 3.10 / 3.19）：

```text
ΔU ≤ λ_max * δ² / 2
δ ≲ (2/5) * r_min
```

`delta_from_energy_error(lambda_max, energy_error)` 与 `max_safe_delta(positions)` 提供这两个换算。

## 3. 为什么不用 ITLEE 判事件新旧

Williams 的 ITLEE 判定的是**局域环境（LE）**的新旧，事件跟着环境走；事件是否真的可迁移，由搬运后的重建校验决定。因此本实现：

- `Event.hash` / `EventInfo.key_rxn` 保持精确图哈希主键；
- ITLEE 用于 LE 分类（`Cluster.is_equivalent`）、匹配校验（`get_match_mode(tolerance=...)`）和重建守卫；
- 不把 δ 塞进事件哈希，不按 ITLEE 静默合并事件键。

## 4. 实现落点

| 文件 | 变更 |
| --- | --- |
| `src/graphatoms/geometry/itlee.py` | ITLEE 核心：指纹、锚点回溯、Procrustes、`ITLEEResult` |
| `src/graphatoms/system/graph.py` | `SysGraph.get_itlee_colors/get_itlee_equiv/is_equivalent` |
| `src/graphatoms/system/bonds.py` | `get_match_mode` / `match` 的 `tolerance` 参数；L2 过滤 |
| `src/graphatoms/enterpoint/network/_equivalence.py` | 每 LE δ 的 `EquivalenceRegistry` |
| `src/graphatoms/enterpoint/network/_clsnetwork.py` | `equivalence.json` 加载/持久化 |
| `src/graphatoms/enterpoint/runner/common/_base.py` | `analysis` 中的 ITLEE LE 去重 |
| `src/graphatoms/enterpoint/runner/common/_helper.py` | `helper_match`/`helper_apply` 传入 δ；`record_reconstruction` |
| `src/graphatoms/enterpoint/runner/general/otfkmc.py` / `rxngen.py` | L3a rmsd 守卫与 δ 折半 |
| `src/graphatoms/enterpoint/config/__init__.py` | `GeometryConfig` |

## 5. 配置

```yaml
geometry:
  tolerance: 0.0        # ITLEE δ, Å (L2); 0 disables
  allow_reflection: false
  use_hash_bucket: false
  verify_graph: true
  reconstruction_check: true
  delta_halving: true
  max_branches: 200000
```

- `use_hash_bucket=false` 用于 LE 容差分类：GraphAtoms 的 3 轮 WL 哈希不是规范型，把它当否决会产生假新。
- `tolerance=0` 时所有旧行为不变。

## 6. 失败模式

| 情况 | 行为 |
| --- | --- |
| ITLEE 搜索超过 `max_branches` | 返回 `None`，发出 `RuntimeWarning`；失败关闭 |
| `δ >= r_min` | 结果标记 `ambiguous=True` 并警告 |
| 图同构无候选 | 返回 `None`（假新，交给探索；不做拓扑无关 fallback） |
| 几何残差 > δ | 丢弃该映射 |
| L3a rmsd > δ/√n | 记录失败，δ 折半（可配置），持久化 registry |
| 旧网络无 `equivalence.json` | 使用空 registry，δ 回退配置值 |

## 7. 测试

`src/tests/test_itlee.py` 覆盖：

- 旋转/平移/置换不变、噪声阈值、颜色、反射；
- `max_safe_delta` / `delta_from_energy_error`；
- `SysGraph` graph/itlee 等价与哈希否决；
- `get_match_mode` 精确路径不变、CN 过期 pattern 的容差匹配、几何拒绝；
- `EquivalenceRegistry` 持久化与折半；
- `_unique_clusters_by_itlee` 的合并与角色保护。

## 8. 后续

- L3b：优化后产物 vs 参考 P 的完整能量/位移/图校验；
- 用 nauty 规范型替换 WL 哈希作为 L0 桶键；
- 位点级速率加权（当前 BKL 仍是事件类型级 `nmatched × rate`）；
- 事件目录 alias 层，仅在 L3 验证通过后消除近重复事件的双重计数。
