# UrbanPhotoMeshQA 项目交接说明

## 1. 项目来源

本项目从冻结的旧项目复制质量评估相关内容而来：

```text
/Users/wangfugui/Paper/三维CIM水印/HKMeshIdentityAuth
```

旧项目同时研究过同源检索、零水印和质量评估。经组会讨论后，新项目只保留质量评估主线。旧目录不得继续修改；其冻结文件清单位于 `docs/migration/`。

## 2. 研究问题

研究对象是香港 Open3Dhk Individualised Building 的摄影测量单体 glTF Mesh。实际推理只输入一个待评估资产包，不依赖其干净原版：

```text
待评估 .gltf + .bin + textures
  → 原生解析
  → Point / Mesh / Morphology / Texture / Local Patch 特征
  → 无参考质量模型
  → OQI、几何质量、纹理质量、类型/强度、低质量Patch
```

Clean/Attacked配对只可在数据集构建时生成客观真值，不可作为Quality Student输入。

## 3. 旧结果如何使用

`artifacts/quality/legacy_baseline` 是旧协议模型和结果。旧纹理攻击发生在渲染图而非源贴图，已在真实贴图模糊测试中失败：轻、中、重均被预测为Clean，纹理质量接近100。因此旧结果只作为对照，不是新项目最终结论。

`artifacts/pretrained_backbone` 保存Point、Mesh及四分支旧权重。当前正式基线冻结 Point 和 Mesh Encoder，新论文不使用“Identity Base”叙事，而是将它们解释为通用几何表征。端到端质量微调是后续提升方向，不是本轮已完成结论。

## 4. 正式数据划分

正式清单：`artifacts/manifests/source_manifest_seed2026.json`。

| Split | 数量 | 图幅 |
|---|---:|---|
| Train | 80 | 11-NE-13B、11-SW-10C、11-SW-15A、11-SW-5A、11-SW-9D |
| Val | 40 | 11-SW-4B |
| Test | 32 | 11-NE-14A、11-SW-14B |
| Blind | 32 | 11-SW-3B、11-SW-4D |

同一建筑的Clean和所有攻击版本只能属于一个Split。

## 5. 第一版攻击协议

六个粗类别，每类三级：

1. `geometry_hole`
2. `mesh_simplification_qem`
3. `geometry_noise_spike`
4. `texture_detail_loss`
5. `texture_region_missing`
6. `texture_misalignment`

`texture_detail_loss`包含两种生成机制：摄影测量相关的高斯模糊和发布相关的贴图降采样。模型只预测粗类别，具体子类型写入metadata并分别报告。每个建筑根据asset_id哈希固定选择一种机制，四个Split内保持均衡。

每个攻击结果必须是可独立打开的 `.gltf + .bin + textures + metadata.json`，并在导出后从磁盘重新读取。详细参数见 `configs/quality_real_gltf_v1.json`。

## 6. 缓存原则

`.npz`只是缓存。缓存命中必须同时满足：

- glTF、BIN和全部贴图组合SHA-256一致；
- 代码、checkpoint、采样点、Patch、渲染参数及seed组成的提取器签名一致；
- 数组keys、shape、dtype和数值一致；
- 神经特征余弦相似度不低于0.999999，最大绝对差不高于1e-5；
- 最终质量预测差异不高于1e-4。

新协议优先采用“先落盘glTF，再重读并只提取一次正式特征”，避免预导出/后导出重复计算。

## 7. 执行约束

- 大规模数据生成、训练和正式推理实验只在租用GPU服务器运行，不在Mac运行。
- Mac只做文件组织、轻量静态检查和结果查看。
- 第一阶段仅3个典型建筑Pilot；QEM无法可靠保留UV时必须停止该类型并报告，不得伪造。
- Pilot通过后才允许全量生成184个建筑的攻击资产并训练。

## 8. 当前交接点

1. 六类真实 glTF 退化、全量缓存、客观真值和单种子训练均已完成。
2. 正式发布包位于 `artifacts/quality/final/release_seed2026_v1/`。
3. 详细数据审计、指标、消融和局限见 `docs/EXECUTION_STATUS_ZH.md`。
4. 下一轮优先改善 Blind 泛化，再增加 UV/图像 Patch 对应的局部纹理质量监督。
