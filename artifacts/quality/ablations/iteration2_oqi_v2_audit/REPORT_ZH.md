# OQI v2机器真值审计

记录数：3493；objective no-op：90。

## 全局分布

| 分数 | 均值 | 中位数 | ≤0.01 | ≥0.99 |
|---|---:|---:|---:|---:|
| geometry_quality | 0.751 | 1.000 | 4.0% | 52.9% |
| texture_quality | 0.846 | 1.000 | 0.0% | 56.5% |
| overall_quality | 0.596 | 0.636 | 4.0% | 9.4% |

## 各攻击Overall均值

| 攻击 | light | medium | heavy |
|---|---:|---:|---:|
| geometry_hole | 0.573 | 0.239 | 0.086 |
| geometry_noise_spike | 0.625 | 0.460 | 0.344 |
| mesh_simplification_qem | 0.795 | 0.640 | 0.494 |
| texture_detail_loss | 0.859 | 0.748 | 0.661 |
| texture_misalignment | 0.935 | 0.844 | 0.730 |
| texture_region_missing | 0.565 | 0.379 | 0.251 |

## 几何分数瓶颈占比

- geometry_fidelity: 51.7%
- completeness: 2.5%
- topology_health: 45.9%

## 自动告警

- geometry_hole分数饱和：低端21.9%，高端0.0%。

## 18组代表案例视觉复核

采用每类攻击三级分数最接近全体中位数的完整建筑序列，生成`representative_18_cases.png`。单一固定视角检查结论：

- 孔洞与纹理区域缺失的三级变化清楚，但孔洞heavy大量接近0，低端区分能力不足；
- QEM简化在全局渲染中变化较弱，当前分数下降可能比可见变化更敏感；
- 几何噪声、纹理细节损失和纹理错位需要多视图及局部放大才能可靠判断，单视图不足以校准感知严重程度；
- 当前六类heavy平均OQI跨度为`0.086–0.730`，不能直接假定全部差异都来自真实感知严重度；
- Pseudo-MOS v3应增加多视图局部感知教师、跨攻击标尺和soft-min候选，并保留原始客观指标供追溯。
