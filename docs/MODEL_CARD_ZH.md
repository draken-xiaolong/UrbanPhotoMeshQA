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
- `release_seed2026_v1` 的权重来自 QEM 重复项正式剔除之前的训练流程；其正式
  指标已在 3493 条清单上重算，但训练集中曾包含两条后来剔除的记录。后续实验
  必须使用 `--require-formal` 并保存数据有序样本指纹。
