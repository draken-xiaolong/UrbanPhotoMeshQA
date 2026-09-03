# UrbanPhotoMeshQA 执行与实验记录

更新日期：2026-09-03；正式随机种子：`2026`。所有攻击生成、特征提取和训练均在 RTX 4090 服务器执行，Mac 未运行实验。

## 1. 项目边界

本项目只研究香港城市摄影测量单体 glTF Mesh 的无参考质量评估，不再开展同源检索、零水印、身份认证或恶意拼接实验。旧项目 `HKMeshIdentityAuth` 已冻结，冻结清单共 8235 个文件，本轮结束时 8235/8235 重新校验 SHA-256 全部匹配。

## 2. 正式数据

- 原始官方单体建筑：184栋；
- 原始划分：Train 80、Val 40、Test 32、Blind 32；
- 每栋：1个 Clean + 6类退化×3个强度；
- 已生成真实攻击 glTF 包：3312个；
- 客观指标计算和初始训练样本：3496条；
- 质检发现3个极简模型的 QEM medium/heavy 文件哈希完全相同，已从正式评测清单剔除 heavy；
- 正式评测清单：3493条，Train 1518、Val 760、Test 607、Blind 608。

六类真实文件级退化为：局部孔洞、QEM简化、几何噪声/尖刺、贴图细节损失、贴图区域缺失、贴图错位/重影。每类均为 light/medium/heavy，且同一建筑的三档固定退化位置，只改变强度。

## 3. 数据和缓存审计

- 184/184个源 glTF 通过本地与服务器双重审计；
- 3312/3312个攻击包通过哈希、依赖、解析、非空和非 no-op 检查；
- 3组几何噪声的相邻等级客观标量并列，文件哈希不同，记为大坐标 float32 量化警告；
- 3组 QEM 完全重复等级按明确规则剔除，未手工挑选其他样本；
- Pilot 54/54 的 NPZ 重提取、写入回读和四分支神经特征审计通过；
- Pilot 54/54 的最终 Quality Head 输出差异最大值为 `0.0`，优于 `1e-4` 验收阈值。

## 4. 模型

```text
待评估 glTF 资产包
  ├─ Point：1024个确定性表面采样点 → 冻结 Point Encoder（256D）
  ├─ Mesh：三角面+邻接+拓扑 → 冻结 Mesh Encoder（256D）
  ├─ Morphology：整体形态不变量（13D）
  └─ Texture：6个确定性多视角渲染 → 冻结 MobileNetV3（576D）
             ↓
       四分支投影+注意力+MLP Quality Head
             ↓
  Overall OQI + Geometry Quality + Texture Quality

独立局部v2：拓扑划分16个Patch → Face可见视图 + UV纹理atlas + 几何描述
             → 每Patch的Geometry、Texture和Overall Quality
```

推理阶段只读取一个待评估 glTF，不使用 clean reference、ICP配准或差值特征。clean/attacked 配对只用于训练真值构建。

## 5. 主模型结果

仅使用 Val OQI SRCC，再用 MAE 打破平局来选模；Test 和 Blind 始终锁定。最终选择四分支全局 Quality Head。

| Split | OQI MAE↓（校准后） | OQI PLCC↑ | OQI SRCC↑ | Geometry SRCC↑ | Texture SRCC↑ |
|---|---:|---:|---:|---:|---:|
| Val | 0.188 | 0.542 | 0.521 | 0.627 | 0.609 |
| Test | 0.184 | 0.591 | 0.557 | 0.493 | 0.500 |
| Blind | 0.196 | 0.461 | 0.444 | 0.509 | 0.587 |

校准是只在 Val 拟合的非负仿射变换，因此不改变 SRCC 排序，也没有使用 Test/Blind 真值拟合分数。

## 6. 逻辑消融

- Point-only：证明只看表面采样不足；
- Point+Mesh+Morphology：验证原生网格与整体形态的增益；
- Four-branch：加入真实贴图外观，作为最终全局模型；
- Four-branch+Patch：用于局部缺陷定位，不强行替换全局 OQI；
- Ridge：Val/Test/Blind OQI SRCC 为 0.513/0.505/0.369，低于非线性四分支模型；
- 去除退化辅助监督：Blind 下降，说明退化辅助任务对质量表征有正向作用。

## 7. 当前限制

1. Blind OQI SRCC `0.444` 属于已验证可行但还不是成熟发表结果；
2. 只使用香港 Individualised Building，尚未验证其他城市对象或真实人工 MOS；
3. 局部v2已增加Face—UV—Texture监督，但Geometry与Overall排序仍为中等水平，需要在新的Train/Val实验版本中继续提高；
4. 当前 OQI 是客观全参考指标合成的训练目标，还不等同于人的主观质量感知。

## 8. 关键入口

- 正式协议：`docs/REAL_GLTF_QUALITY_PROTOCOL_ZH.md`
- 单模型推理：`scripts/infer_real_gltf_quality.py`
- 客观真值：`scripts/build_objective_quality_targets_real.py`
- 质量模型训练：`scripts/train_real_gltf_quality.py`
- 缓存与最终输出等价审计：`scripts/validate_cache_neural_equivalence.py`
- 局部v2特征：`scripts/extract_local_patch_features.py`
- 局部v2训练：`scripts/train_local_patch_quality.py`
- 局部v2缓存/在线等价审计：`scripts/audit_local_cache_equivalence.py`

## 8.1 局部质量v2（已冻结）

局部v2只用Train/Val选择，Checkpoint SHA256为`3a94425fc791987db1d1c7cf3843a4be15e6acb48a3a7629c16ddf9bf38bb8ef`。冻结后一次性解锁Test/Blind，未再调参：

| Split | Geometry SRCC | Texture SRCC | Overall SRCC |
|---|---:|---:|---:|
| Val | 0.381 | 0.602 | 0.422 |
| Test | 0.433 | 0.623 | 0.446 |
| Blind | 0.468 | 0.578 | 0.432 |

19条同一建筑Clean及全部攻击记录的缓存/单glTF在线推理完全一致，三输出最大绝对差为`0.0`。发布目录为`artifacts/quality/final/local_patch_v2_seed2026/`。

## 9. 冻结 Base 泛化筛选（已完成）

- 环境审计：`scripts/audit_gpu_quality_environment.py`；
- 四组冻结 Base 泛化筛选：`scripts/run_quality_generalization_minimal.py`；
- Val-only 汇总与 promotion gate：`scripts/summarize_quality_generalization_val.py`；
- 固定配置：`configs/quality_generalization_minimal_seed2026.json`；
- 运行说明：`docs/GENERALIZATION_EXPERIMENT_RUNBOOK_ZH.md`。

候选训练只加载 Train/Val NPZ；Test/Blind 在唯一候选冻结前不加载、不评测。部分解冻与
端到端微调需要接入原始缓存和实际 Encoder 反向传播，不能由冻结特征训练替代，因此仅在
第一阶段未达到 Val promotion gate 时进入第二阶段。

seed=2026 的 Val-only 结果如下：

| 候选 | OQI SRCC | Geometry SRCC | Texture SRCC | 相对release晋级 |
|---|---:|---:|---:|---:|
| release_seed2026_v1 | 0.520 | 0.626 | 0.593 | reference |
| B0 mean/std | 0.500 | 0.604 | 0.605 | 否 |
| B1 robust norm | 0.490 | 0.644 | 0.587 | 否 |
| B2 robust + tile balance | 0.502 | 0.586 | 0.639 | 否 |
| B3 robust + tile + worst | 0.549 | 0.586 | 0.653 | 否（几何下降0.040） |
| B4 release warm-start | 0.493 | 0.632 | 0.594 | 否 |

B3证明图幅鲁棒损失能提高整体和纹理排序，但相对正式release的几何SRCC下降超过0.02
门限；B4保住几何却没有保住整体排序。因此保持原release，不解锁Test/Blind，下一阶段
进入真实Encoder部分解冻，而不是继续搜索冻结Head超参数。
