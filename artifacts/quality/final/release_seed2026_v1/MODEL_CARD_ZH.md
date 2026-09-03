# UrbanPhotoMeshQA 模型卡

## 用途

输入一个完整的香港 Individualised `.gltf + .bin + textures` 建筑资产包，在无原始参考模型的条件下预测 Overall OQI、几何质量、纹理质量和局部几何 Patch 质量。

## 发布组成

- `global_quality_head.pt`：全局四分支 Quality Head，按 Val OQI SRCC 选择；
- `local_patch_head.pt`：独立局部 Patch Head，按 Val 局部 MAE 选择；
- `calibration.json`：仅使用 Val 拟合的 OQI/几何/纹理仿射校准；
- 冻结 Point 与 Mesh checkpoint 位于 `artifacts/pretrained_backbone/`。

## 注意

- OQI 越高代表质量越好，输出范围 `[0,1]`，界面可乘100显示。
- 模型只在当前香港建筑数据域内验证，不应直接宣称对所有 Mesh 或所有摄影测量场景泛化。
- 局部 Patch 输出目前主要针对几何缺陷，不应解读为像素级纹理缺陷定位。
- 本版 Checkpoint 的训练流程早于三条重复 QEM 记录的正式剔除；正式指标已按
  3493 条清单重算，但训练集中曾包含其中两条。它作为历史基准保留，详情见
  `PROVENANCE_AUDIT_ZH.md`。后续模型必须严格使用 1518 条正式 Train。
