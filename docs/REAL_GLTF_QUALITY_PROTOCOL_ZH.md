# 三维 Mesh 质量评估真实 glTF 数据协议迭代文档

> 状态：第一版已执行
> 日期：2026-09-02
> 原则：本文档只定义下次迭代，当前不生成新数据、不重新训练、不覆盖现有结果。

## 1. 迭代目标

质量评估必须模仿真实使用流程：

```text
获取一个待评估 glTF 资产包
  → 原生解析 .gltf + .bin + textures
  → 对单个模型独立做尺度/坐标归一化
  → 提取 Point、Mesh、Morphology、Texture 和 Patch 特征
  → 无参考 Quality Model
  → 输出总质量、分项质量、退化类型、强度和低质量区域
```

训练、验证和测试中的每一条受攻击记录，都必须对应一个真实落盘、可独立打开的 glTF 资产包，不再只保存 `.npz` 特征或在内存中临时攻击。

## 2. “一个 glTF 样本”的定义

`.gltf` 是入口文件，完整样本是一个资产包：

```text
sample_id/
├── model.gltf
├── model.bin
├── textures/
│   ├── texture_001.jpg
│   └── texture_002.png
└── metadata.json
```

必须满足：

1. `model.gltf` 中的 buffer 和 image URI 均为包内相对路径。
2. 离开原数据目录后仍可在 MeshLab、Blender 或解析程序中打开。
3. 攻击后的 `.gltf` 必须被重新读取并校验，不能直接使用攻击函数的内存输出训练。
4. `metadata.json` 记录源模型ID、攻击类型、物理参数、强度级别、随机种子、生成器版本和文件校验和。

## 3. 当前数据集协议的问题

### 3.1 纹理攻击位置不对

当前 `build_texture_quality_targets.py` 的实际过程是：

```text
清晰 glTF → 渲染六视图 → 对二维图做 JPEG/模糊/亮度/下采样/遮挡
```

这个协议学到的是截图或相机画面退化，不是 glTF 资产自身的贴图质量。新协议应改为：

```text
清晰 glTF → 改写贴图文件 → 导出受攻击 glTF 资产包
            → 重新读取该 glTF → 渲染/特征提取 → 质量真值与无参考训练
```

本次高层建筑样例已暴露该域差异：源贴图模糊后，现有模型仍把轻、中、重三档全部识别为 `clean`，纹理质量均接近100分。

### 3.2 几何攻击没有作为可追溯 glTF 保存

`build_geometry_quality_targets.py` 和部分特征评估代码会对干净模型调用 `apply_mesh_attack`，随后直接在内存中采样和计算指标。问题是：

- 没有对应的攻击后 `.gltf`，无法人工核查。
- 没有经过“导出→重新解析”，无法暴露索引、材质、UV、法向和缓冲区错误。
- 训练数据路径与真实部署时的文件路径不一致。

新协议要求所有几何攻击也先导出为 glTF 资产包，然后从磁盘重新读取。

### 3.3 模型质量与观测条件混在一起

以下变化不是 glTF 模型本身的质量缺陷：

- 相机视角抖动；
- 渲染背景替换；
- 对最终截图做遮挡；
- 对最终截图做运动模糊或 JPEG 压缩。

这些适合“视图/重拍鲁棒性”测试，不应直接作为“单个 glTF 资产的内在质量”训练数据。

处理原则：

- 贴图局部涂抹/缺失：正式纳入下一版纹理质量攻击，必须真正改写源贴图并导出 glTF。
- 相机或截图遮挡：放入独立的观测鲁棒性数据集。
- 仿真拍照后二次重建：只有最终生成了真实重建 Mesh，并导出为 glTF 资产包，才能进入 glTF 质量数据集。

### 3.4 纹理攻击强度不完整

旧协议中多个纹理攻击只有一个固定参数，例如 `blur1.5`、`jpeg50`和 `brightness0.55`。这不足以学习质量的单调变化。

新协议中每种退化至少应有轻、中、重三档，并保存原始物理参数，不只保存一个模糊的 `severity=0.5`。

### 3.5 质量真值的定义需要区分维度

“重新三角化”在当前实现中不改变表面外形和纹理，只增加顶点、三角面并改写连接拓扑。它不属于直观的模型保真度损失，本轮迭代将其从质量数据集中移除，不参与 Quality Student 和 Overall OQI 训练。该攻击只保留在旧项目的历史身份鲁棒性结果中，不迁入新质量评估项目。

建议保留分项输出：

```text
Geometry Fidelity  几何保真度
Texture Fidelity   纹理保真度
Topology Health    拓扑健康度
Completeness       完整度
Overall OQI        经明确权重或学习校准得到的总分
```

总分不应掩盖分项结果。

### 3.6 “对齐”不能暗中依赖原始模型

真实无参考场景只有一个待评估 glTF，因此推理阶段只允许：

- 基于待评估模型自身的中心化、尺度归一化和确定性坐标规范化；
- 基于训练集统计量的特征标准化。

不允许用原始干净模型做 ICP 配准、尺度对齐或差异特征计算。干净模型只能在数据集构建阶段生成客观真值，不得作为学生模型的输入。

## 4. 建议的真实 glTF 数据集结构

```text
quality_real_gltf_v1/
├── clean_reference/
│   └── <asset_id>/model.gltf + model.bin + textures/
├── attacked/
│   └── <split>/<asset_id>/<attack>/<level>/
│       ├── model.gltf
│       ├── model.bin
│       ├── textures/
│       └── metadata.json
├── manifests/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── blind.jsonl
├── targets/
│   └── objective_quality_targets.jsonl
└── audit/
    ├── parse_validation.json
    ├── split_validation.json
    └── contact_sheets/
```

实际外置数据根目录统一为：

```text
/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data/
├── HK3D-Individualised/
├── HK3D-Individualised-Attack/
├── HK3D-Individualised-Attack-Samples/
├── Feature-Cache/
├── Manifests/
└── Audit/
```

各目录职责如下：

- `HK3D-Individualised/`：只保存本项目正式 Train/Val/Test/Blind 使用的原始单体子集，不复制整个外部源数据盘。
- `HK3D-Individualised-Attack/`：保存所有正式生成的受攻击 glTF 资产包，是质量训练数据的唯一攻击实体来源。
- `HK3D-Individualised-Attack-Samples/`：保存供人工查看的少量典型示例，不直接作为正式清单的替代品。
- `Feature-Cache/`：保存从 clean/attacked glTF 提取的 `.npz` 加速缓存。
- `Manifests/`：保存184个单体的划分、路径、攻击参数和缓存索引。
- `Audit/`：保存解析校验、哈希、缓存一致性和人工预览结果。

特征 `.npz` 只是可重建的加速缓存，不再是数据集的唯一实体。每条 `.npz` 记录必须能反查到其 glTF 相对路径和校验和。

## 5. 第一版精简攻击集（已执行）

为了先快速验证真实 glTF 无参考质量评估，第一版不追求攻击类型大而全，只保留 `3种几何 + 3种纹理`，每种统一为 `light / medium / heavy` 三档。

```text
clean + 6种攻击 × 3种强度 = 每个原始模型19个版本
```

### 5.1 几何退化（3种）

| 类型 | 真实场景 | 轻度 | 中度 | 重度 |
|---|---|---:|---:|---:|
| `geometry_hole` 局部孔洞 | 遮挡、反光、采集覆盖不足导致局部无法重建 | 删除5% | 删除15% | 删除30% |
| `mesh_simplification_qem` QEM简化 | 网页/移动端发布、LOD、存储和渲染优化中过度降面 | 保留85%面 | 保留70%面 | 保留55%面 |
| `geometry_noise_spike` 表面噪声/局部尖刺 | 多视图匹配错误造成墙面起伏、边缘融化、局部异常顶点 | 对角线0.1% | 0.3% | 0.6% |

`geometry_noise_spike` 第一版作为一个统一类别，只用同一套局部法向位移生成器，不在起步阶段再拆成全局噪声、尖刺和漂浮面多个分类。

### 5.2 源贴图退化（3种）

| 类型 | 真实场景 | 轻度 | 中度 | 重度 |
|---|---|---:|---:|---:|
| `texture_detail_loss` 低清晰度贴图 | 航片分辨率不足、对焦/融合模糊或发布降采样 | 同一建筑固定使用模糊0.8或降采样1/2 | 模糊2.0或降采1/4 | 模糊4.0或降采1/8 |
| `texture_region_missing` 局部贴图缺失 | 遮挡、影像覆盖不足、纹理拼接空白 | 缺失5% | 缺失15% | 缺失30% |
| `texture_misalignment` 纹理错位/重影 | 相机配准误差、照片重投影错误、纹理来源切换 | 局部小偏移 | 中等偏移/轻重影 | 大偏移/明显重影 |

每种贴图攻击要对 glTF 引用的真实 `.jpg/.png` 文件执行，之后导出和重新打开 glTF。

`texture_region_missing` 的缺失区域需要记录掩码、实际缺失像素比例和随机种子。攻击后应保证几何、拓扑、UV 和材质索引不变，只降低纹理完整性。

第一版的 `texture_misalignment` 通过源贴图局部平移和透明叠加复现错位/重影，不改写 UV 坐标。暂不单独训练 JPEG、全局亮度下降、局部曝光接缝、纹理拉伸、材质丢失和 UV 坐标扰动等更细子类。等六类基线稳定后，根据混淆矩阵逐项增加，不一次性扩充。

### 5.3 暂不混入的观测攻击

- 背景变换、视点抖动、截图遮挡、相机失焦、传感器噪声。

这些攻击不纳入新质量评估项目。只有它们真正生成二次重建 glTF 时，才以“重建模型质量”的形式重新评估是否纳入。

## 6. 数据划分与防泄漏

1. 先按原始单体 `asset_id` 划分，再生成攻击。
2. 同一建筑的 clean 和所有攻击版本必须只属于一个 split。
3. 保留现有的跨图幅验证思路，不仅随机拆分同一图幅内的建筑。
4. 攻击类型和强度在 train/val/test/blind 中平衡，但 blind 的原始模型不参与训练。
5. 增设“未见攻击参数”测试：训练使用若干参数，测试使用中间值，检查质量连续性。

### 6.1 本项目应复制的原始数据范围

当前四分支 Base 的正式源清单为：

```text
artifacts/base/four_branch_fusion_seed2026_v1/source_manifest.json
```

该清单共 `184` 个单体建筑：

| Split | 单体数 | 图幅 |
|---|---:|---|
| Train | 80 | `11-NE-13B`, `11-SW-10C`, `11-SW-15A`, `11-SW-5A`, `11-SW-9D` |
| Val | 40 | `11-SW-4B` |
| Test | 32 | `11-NE-14A`, `11-SW-14B` |
| Blind | 32 | `11-SW-3B`, `11-SW-4D` |

第一步只按该184条记录从原始 `/Volumes/SANDISK-ELE/HK3D-Individualised` 复制资产目录。`cross_format_source_seed2026` 的72个样本已是这个集合的子集，不重复复制。

原始目录层级保持不变，不在路径中额外嵌套 split；split 只写入 manifest：

```text
/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data/HK3D-Individualised/
└── <图幅>/BUILDING/<asset_id>/
    ├── <asset_id>.gltf
    ├── <asset_id>.bin
    └── <asset_id>*.jpg/png
```

攻击数据同样保留“图幅→对象类型→单体”的定位方式，再在单体下增加攻击和强度：

```text
/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data/HK3D-Individualised-Attack/
└── <图幅>/BUILDING/<asset_id>/<attack>/<level>/
    ├── <asset_id>.gltf
    ├── <asset_id>.bin
    ├── textures/
    └── metadata.json
```

现有 `/Volumes/SANDISK-ELE/HK3D-Individualised-Attack-Samples` 后续整体迁入 `UrbanPhotoMeshQA-Data/HK3D-Individualised-Attack-Samples/`，但当前迭代计划阶段不移动。

### 6.2 `.npz` 缓存复用规则

最稳妥的正式数据链路是：

```text
生成攻击 → 导出 glTF 资产包 → 从磁盘重新打开 glTF
         → 只提取一次特征 → 保存 `.npz`
```

新版默认不在 glTF 导出前提取正式特征，因此不存在重复提取。如果攻击生成阶段已产生内存 `.npz` 或需要复用旧缓存，必须同时通过以下条件：

1. 计算资产指纹 `asset_digest`：按 glTF URI 顺序组合 `.gltf + .bin + 全部贴图` 的 SHA-256。
2. 计算提取器签名 `extractor_signature`：代码版本、Base checkpoint SHA-256、采样点数、Patch数、渲染分辨率和随机种子的组合。
3. `.npz` 的两个签名必须与当前 glTF 和提取器完全相同。
4. 对缓存数组而不是 `.npz` 压缩文件字节做一致性检查：keys、shape、dtype必须一致；确定性原始描述量需 `allclose(rtol=1e-6, atol=1e-7)`；神经特征需同时满足余弦相似度 `>=0.999999` 和最大绝对差 `<=1e-5`。
5. 最终 Quality 输出差异不得超过 `1e-4`。

只有上述条件全部通过才可命中缓存。旧 `.npz` 如果没有资产指纹和提取器签名，默认不可直接复用；任一检查失败，必须从导出后的 glTF 重新提取。

## 7. 真值与训练输入的边界

数据集构建时可以使用 clean/attacked 配对计算真值，例如 Chamfer、法向误差、缺失比例、视图 SSIM 和边缘差异。但训练学生模型和实际推理时：

```text
允许输入：attacked/model.gltf 资产包
不允许输入：clean reference、clean/attacked 差值、攻击类型标签、攻击强度标签
```

攻击类型和强度是监督信号和评估真值，不是推理输入。

## 8. 每个样本的自动审计

数据生成后、特征提取前必须通过：

- glTF JSON 可解析；
- 所有 `.bin` 和贴图 URI 存在；
- 重新读取后顶点数和面数非零；
- 坐标、法向和 UV 无 NaN/Inf；
- 材质索引有效；
- 贴图攻击不得意外改变几何哈希；
- 几何攻击必须记录面数、顶点数、边界边、非流形边和连通分量变化；
- clean/light/medium/heavy 的客观退化总体上应单调；
- 每类攻击随机抽样生成六视图 contact sheet 供人工复核。

## 9. 新协议的验收条件

在重新训练前，数据集必须先满足：

1. 攻击记录与 glTF 资产包一对一，无悬空记录。
2. 所有样本可由正式推理入口 `infer_mesh_quality.py` 直接读取。
3. 清晰和受攻击模型的差异来自资产文件，而不是特征提取后临时注入。
4. 训练、验证、测试和 blind 无 `asset_id` 及图幅泄漏。
5. 至少在当前高层建筑样例上，源贴图模糊的预测质量能随 `light → medium → heavy` 单调下降。
6. 报告同时给出 Overall OQI 和 Geometry/Texture/Topology/Completeness 分项，不只报一个总分。

## 10. 后续可扩展问题

1. 本论文的质量定义是主要聚焦“与原始资产的保真度”，还是同时覆盖“拓扑健康/工程可用性”？
2. 第一版按六种攻击完成基线后，优先根据哪个混淆类型扩展 JPEG、曝光接缝、纹理拉伸或材质丢失？
3. 仿真拍照后二次重建是否能产生足够真实且可重复的 glTF 质量样本，并在后续阶段作为扩展实验？
4. Overall OQI 的分项权重是由工程规则指定，还是使用主观质量标注学习？
5. 第一版是否只使用香港 Individualised Building glTF，待协议稳定后再扩展其他对象或格式？

## 11. 建议的后续执行顺序（当前不执行）

```text
冻结现有结果
  → 确定质量定义和攻击清单
  → 实现可追溯 glTF 攻击导出器
  → 小样本人工审核
  → 全量生成并重新解析
  → 从 glTF 提取 Base/Patch 特征
  → 重建客观真值
  → 重训 Quality Student 和 OQI Head
  → 文件级无参考测试
  → 与旧协议做对照消融
```

## 12. 单一质量评估新项目迁移方案（当前不执行）

### 12.1 项目定位与命名

新项目英文名定为：

```text
UrbanPhotoMeshQA
```

含义是 `Urban Photogrammetric Mesh Quality Assessment`，即“城市摄影测量三维 Mesh 质量评估”。目标项目路径为：

```text
/Users/wangfugui/三维实景模型质量评估/UrbanPhotoMeshQA
```

外置数据根目录从原计划的 `HKMeshIdentityAuth-Data` 改为：

```text
/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data
```

新项目的唯一研究主题是：

> 面向香港城市摄影测量单体 glTF Mesh 的文件级、无参考、全局—局部质量评估。

新项目不再将同源检索和零水印作为下游任务，不再以“一个通用身份 Base 支持三个下游”为论文叙事。

### 12.2 旧项目冻结原则

旧目录：

```text
/Users/wangfugui/Paper/三维CIM水印/HKMeshIdentityAuth
```

执行迁移时先生成冻结清单，记录文件路径、大小、修改时间和 SHA-256。为保证“冻结”真正可追溯，实际操作应当是：

```text
旧项目保留不变 + 质量相关内容复制到新项目
```

而不是从旧目录删除或剪切文件。这样才不会破坏已有实验的相对路径、历史记录和复现能力。新项目验收完成后，旧项目只读保留，不再迭代。

### 12.3 迁入的内容

只迁入完成质量评估必须的最小自洽集合：

```text
UrbanPhotoMeshQA/
├── README.md
├── pyproject.toml / requirements
├── src/urbanphotomeshqa/
│   ├── io/              # glTF/GLB/OBJ读取，主协议优先glTF
│   ├── attacks/         # 六种真实glTF质量退化
│   ├── features/        # Point/Mesh/Morphology/Texture/Patch
│   ├── models/          # 质量编码器、Student、OQI和Patch Head
│   └── metrics/         # 客观真值与评估指标
├── scripts/
│   ├── prepare_subset.py
│   ├── export_real_gltf_attacks.py
│   ├── audit_gltf_dataset.py
│   ├── extract_quality_features.py
│   ├── train_quality_model.py
│   └── infer_mesh_quality.py
├── configs/
├── tests/
├── docs/
└── artifacts/
    ├── pretrained_backbone/
    ├── quality/
    └── audits/
```

具体复用内容包括：

- glTF/GLB/OBJ 解析、表面采样、Mesh面图、Morphology、纹理渲染和Patch特征代码；
- 质量退化生成、几何/纹理客观真值、Quality Student、OQI Head、局部Patch质量定位代码；
- 当前最终质量相关checkpoint、results、训练配置和评估报告；
- 新迭代需要的预训练编码器权重，在新项目统一命名为 `pretrained_backbone`，不再延续“Identity Base”的叙事。

### 12.4 不迁入的内容

- 同源检索排序、局部Patch身份匹配和R@1实验；
- 零水印生成、NC协议、攻击后水印验证；
- 恶意拼接侵权判定；
- 仅为身份鲁棒性服务的截图、背景、相机抖动和重新三角化数据；
- `artifacts/retrieval`、`artifacts/zero_watermark` 及旧研究方向的Rubbish历史实验。

如果某个公共模块同时被身份和质量脚本引用，只复制其质量评估需要的最小函数并重命名，不将整个旧模块无差别搬迁。

### 12.5 新项目必须创建的交接文档

真正执行时，新项目根目录必须包含：

1. `README.md`：项目定位、数据路径、环境、快速开始和当前基线。
2. `docs/PROJECT_HANDOFF_ZH.md`：旧项目来源、已完成结果、已知问题、六类新攻击、184个源模型划分和下一步执行顺序。
3. `docs/REAL_GLTF_QUALITY_PROTOCOL_ZH.md`：本文档经最终对齐后的纯质量评估版协议。
4. `docs/MIGRATION_MANIFEST.json`：每个迁入文件的旧路径、新路径、SHA-256和迁入原因。
5. `NEW_CHAT_PROMPT_ZH.md`：开启新 Codex 项目对话时使用的提示词。

### 12.6 新对话提示词草案

真正迁移时将以下内容写入 `NEW_CHAT_PROMPT_ZH.md`：

```text
请接续 UrbanPhotoMeshQA 项目。这是一个只研究城市摄影测量单体 Mesh 质量评估的独立项目，不再开展同源检索、零水印或恶意拼接任务。

请先完整阅读 README.md、docs/PROJECT_HANDOFF_ZH.md 和 docs/REAL_GLTF_QUALITY_PROTOCOL_ZH.md，再检查代码、数据manifest、checkpoint和已知基线，不要从头猜测项目状态。

核心目标是：构建面向香港 Individualised Building glTF 的文件级无参考质量评估方法，输出 Overall OQI、几何质量、纹理质量、退化类型/强度，并用局部 Mesh Patch 定位低质量区域。

第一版只使用6类真实资产退化：geometry_hole、mesh_simplification_qem、geometry_noise_spike、texture_detail_loss、texture_region_missing、texture_misalignment；每类只有light/medium/heavy三档。所有训练、验证、测试和Blind攻击样本必须有可独立打开的.gltf + .bin + textures资产包，必须先导出、再从磁盘重新读取，不得用仅存在于内存或二维渲染图上的攻击替代。

外置数据根目录是 /Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data。原始子集只包含正式manifest中的184个建筑：Train 80、Val 40、Test 32、Blind 32。不得将整个 /Volumes/SANDISK-ELE/HK3D-Individualised 复制到新数据目录。

无参考推理只能输入待评估glTF；clean模型只可在数据构建时生成质量真值，不得输入Quality Student。.npz只是缓存，只有asset_digest、extractor_signature、数组一致性和最终预测一致性全部通过时才可复用。

所有大规模训练和数据生成必须在租用GPU服务器上运行，不得在Mac上跑实验。开始实施前，先做小样本导出、人工预览、glTF重读和缓存等价性验证；未通过数据协议验收之前不得全量训练。
```

### 12.7 迁移验收条件

1. 旧 `HKMeshIdentityAuth` 的冻结清单和实际文件 SHA-256 匹配，旧项目未被删改。
2. 新项目不包含同源检索和零水印的执行脚本或结果目录。
3. 新项目在不引用旧项目绝对路径的情况下，可独立完成一个 glTF 的质量推理。
4. 新项目文档、checkpoint、manifest和外置数据路径全部可解析。
5. 迁移前后同一质量样本的推理结果数值一致。
