# UrbanPhotoMeshQA

面向香港城市摄影测量单体 glTF Mesh 的文件级、无参考、全局—局部质量评估项目。

## 唯一研究任务

输入一个待评估的 `.gltf + .bin + textures` 资产包，输出：

- Overall OQI（0–100）；
- 几何质量与纹理质量；
- 退化类型与强度；
- 16个局部 Mesh Patch 的质量及低质量区域。

本项目不开展同源检索、零水印、恶意拼接或身份认证任务。

## 第一版真实 glTF 退化

- `geometry_hole`
- `mesh_simplification_qem`
- `geometry_noise_spike`
- `texture_detail_loss`：按建筑确定性选择 `gaussian_blur` 或 `texture_downsample` 子类型
- `texture_region_missing`
- `texture_misalignment`

每类统一为 `light / medium / heavy`。每个源模型包含1个Clean和18个攻击版本。

## 数据

外置数据根目录：

```text
/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data
```

正式源清单共184个建筑：Train 80、Val 40、Test 32、Blind 32。不要复制整个原始 `HK3D-Individualised` 数据集。

## 当前结果（单种子 `2026`）

- 已生成 `184 Clean + 3312 退化 = 3496`个真实文件级 glTF 样本；
- 每个攻击样本均是可独立重载的 `.gltf + .bin + textures + metadata.json`；
- 数据划分为 Train 1520、Val 760、Test 608、Blind 608，同一建筑不跨划分；
- 主模型是冻结 Point、Mesh、Morphology、Texture Base 上的四分支 MLP Quality Head；
- 仅用 Val 选模，Test/Blind 锁定；
- OQI SRCC：Val `0.521`、Test `0.560`、Blind `0.444`；
- Val 仿射校准后 OQI MAE：Val `0.188`、Test `0.185`、Blind `0.196`；
- 54/54 Pilot 的缓存与重新解析 glTF 在 Base 特征和最终 Quality 输出上均通过数值等价审计；
- `scripts/infer_real_gltf_quality.py` 已实现单模型无参考推理，不读取 clean 原模型或差值。

详细交接见 [PROJECT_HANDOFF_ZH.md](docs/PROJECT_HANDOFF_ZH.md)，正式协议见 [REAL_GLTF_QUALITY_PROTOCOL_ZH.md](docs/REAL_GLTF_QUALITY_PROTOCOL_ZH.md)，执行与实验记录见 [EXECUTION_STATUS_ZH.md](docs/EXECUTION_STATUS_ZH.md)。

GPU 服务器与本地通过 Git/Git LFS 同步代码、Checkpoint 和实验结果；原始 glTF
数据保留在各机器的数据盘。具体流程见 [GPU_GIT_WORKFLOW_ZH.md](docs/GPU_GIT_WORKFLOW_ZH.md)。
