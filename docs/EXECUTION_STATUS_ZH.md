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

另一个独立 Patch Head：16×32个局部表面点 → 16个局部几何质量分数
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
3. Patch Head 第一版主要学习几何局部质量，贴图局部热力图需在下一轮增加 UV/图像 Patch 对应监督；
4. 当前 OQI 是客观全参考指标合成的训练目标，还不等同于人的主观质量感知。

## 8. 关键入口

- 正式协议：`docs/REAL_GLTF_QUALITY_PROTOCOL_ZH.md`
- 单模型推理：`scripts/infer_real_gltf_quality.py`
- 客观真值：`scripts/build_objective_quality_targets_real.py`
- 质量模型训练：`scripts/train_real_gltf_quality.py`
- 缓存与最终输出等价审计：`scripts/validate_cache_neural_equivalence.py`
