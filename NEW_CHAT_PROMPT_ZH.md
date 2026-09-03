# 新Codex项目对话提示词

请接续 `UrbanPhotoMeshQA` 项目。该项目只研究城市摄影测量单体Mesh质量评估，不再开展同源检索、零水印、恶意拼接或身份认证任务。

开始工作前，请完整阅读：

1. `README.md`
2. `docs/PROJECT_HANDOFF_ZH.md`
3. `docs/REAL_GLTF_QUALITY_PROTOCOL_ZH.md`
4. `docs/EXECUTION_STATUS_ZH.md`
5. `docs/MODEL_CARD_ZH.md`
6. `configs/quality_real_gltf_v1.json`

随后检查代码、manifest、checkpoint和现有旧协议基线，不要从头猜测项目状态。

核心目标是构建面向香港Individualised Building glTF的文件级无参考质量评估方法，输出Overall OQI、几何质量、纹理质量、退化类型/强度，并用16个局部Mesh Patch定位低质量区域。

第一版只有六个粗类别：

```text
geometry_hole
mesh_simplification_qem
geometry_noise_spike
texture_detail_loss
texture_region_missing
texture_misalignment
```

每类只有light、medium、heavy三级。`texture_detail_loss`包含`gaussian_blur`和`texture_downsample`两个生成子类型，但模型只预测一个粗类别，metadata和评估结果需分别记录子类型。

所有Train、Val、Test和Blind攻击样本必须有可独立打开的`.gltf + .bin + textures + metadata.json`，必须先落盘、再从磁盘重新读取。不得用仅存在于内存或二维渲染图上的攻击替代真实模型资产攻击。

外置数据根目录是：

```text
/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data
```

原始子集只包含正式manifest中的184个建筑：Train 80、Val 40、Test 32、Blind 32。不得复制整个`/Volumes/SANDISK-ELE/HK3D-Individualised`。

无参考推理只能输入待评估glTF。Clean模型只可用于离线生成质量真值，不得输入Quality Student。`.npz`只有在asset digest、extractor signature、数组一致性和最终预测一致性全部通过时才可复用。

所有大规模训练和数据生成必须在租用GPU服务器运行，不得在Mac运行。当前3312个攻击包、3496个初始样本、客观真值、四分支质量模型和完整审计均已完成。不要重复生成或重训，先读取 `artifacts/quality/final/release_seed2026_v1/` 和正式 manifest。下一步优先解决 Blind 泛化和局部纹理 Patch 监督，任何新模型仍只能用 Val 选择，Test/Blind 保持锁定。
