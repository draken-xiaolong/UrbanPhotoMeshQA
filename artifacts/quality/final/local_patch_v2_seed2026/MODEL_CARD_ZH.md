# UrbanPhotoMeshQA 局部质量模型 v2

该模型把单个glTF Mesh划分为16个拓扑Patch，并分别输出`Geometry Quality`、`Texture Quality`和二者最小值定义的`Overall Quality`。分数范围为`[0,1]`，越高越好。

推理只需一个完整的`.gltf + .bin + textures`资产包，不读取Clean参考、攻击类型、攻击强度、模型差值或ICP结果。入口为：

```bash
python scripts/infer_real_gltf_quality.py \
  --gltf /path/to/model.gltf \
  --local-v2-checkpoint artifacts/quality/final/local_patch_v2_seed2026/local_patch_head.pt
```

模型使用共享边拓扑和面积配额区域生长构建Patch；纹理特征同时来自六视图可见面与Face—UV纹理atlas。Clean/Attacked配对仅用于离线生成训练真值。

所有候选只使用Train/Val，最终Checkpoint冻结后才一次性评测Test/Blind。Checkpoint SHA256为`3a94425fc791987db1d1c7cf3843a4be15e6acb48a3a7629c16ddf9bf38bb8ef`，正式3493条样本有序键指纹为`065fe44a9eff3d7410fde9e9dfce1a86041abbf99600b3c4969516bc2c5f1166`。

局部纹理排序已表现出较稳定的跨图幅能力；局部几何分支仍属中等水平，尤其不应把Patch分数解释为精确的缺陷边界或人工MOS。模型目前只在香港Individualised Building数据域和合成文件级退化上验证。
