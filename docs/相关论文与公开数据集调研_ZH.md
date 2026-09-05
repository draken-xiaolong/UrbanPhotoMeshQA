# 三维Mesh质量评价相关论文与公开数据集调研

> 项目：UrbanPhotoMeshQA  
> 调研日期：2026-09-05  
> 研究目标：面向香港城市摄影测量单体glTF Mesh的无参考质量评价

## 1. 本项目的筛选标准

本调研优先收录满足以下条件的工作：

1. 研究对象是真正的三角Mesh、彩色Mesh或纹理Mesh；
2. 评价目标与人类感知质量、MOS或DMOS有关；
3. 论文发表于CVPR、ACM TOG、IEEE TCSVT、ACM TOMM等可信会议或期刊；
4. 有官方论文、代码或公开数据集网址；
5. 对本项目的“几何＋纹理、无参考、局部Patch、跨区域泛化”至少有一项直接启发。

需要严格区分：

- **全参考（FR）**：推理时同时输入Clean参考模型和待评估模型；
- **无参考（NR）**：推理时只输入一个待评估模型；
- **Mesh方法**：直接使用三角面、拓扑、UV或纹理；
- **点云方法**：输入点坐标/颜色，不能直接等同于Mesh方法；
- **真实MOS**：多人主观评分的平均值；
- **Pseudo-MOS**：算法预测的机器分数，不是真实人工MOS。

## 2. 按与本项目相关性排序

### 2.1 第一优先级：HybridMQA（CVPR 2025）

**论文**：HybridMQA: Exploring Geometry-Texture Interactions for Colored Mesh Quality Assessment  
**论文地址**：<https://openaccess.thecvf.com/content/CVPR2025/papers/Sarvestani_HybridMQA_Exploring_Geometry-Texture_Interactions_for_Colored_Mesh_Quality_Assessment_CVPR_2025_paper.pdf>  
**官方代码**：<https://github.com/arshafiee/hybridmqa>

基本属性：

| 项目 | 内容 |
|---|---|
| 对象 | 彩色/纹理Mesh |
| 参考类型 | **全参考** |
| 输入 | Clean参考Mesh＋退化Mesh |
| 输出 | 整体质量分数 |
| 真值 | 公开数据集人工MOS |
| 核心结构 | Mesh图学习＋二维彩色投影＋几何/纹理Cross-Attention |

核心思想是先从Mesh提取三维几何特征，再把这些特征投影到二维图像平面，使几何特征与彩色渲染像素对齐，之后用Cross-Attention建模几何和纹理之间的相互作用。论文在YN2023、SJTU-TMQA、TSMD和VCMesh四个彩色Mesh数据库上测试。

与本项目最相关的部分：

- 图神经网络提取三角面或局部结构信息；
- 几何特征与纹理投影进行空间对齐；
- Cross-Attention融合几何和纹理；
- 使用MAE＋排序损失共同训练；
- 按源模型隔离进行交叉验证，防止同一内容泄漏。

不能直接照搬的部分：HybridMQA推理需要Clean参考Mesh，而UrbanPhotoMeshQA实际推理只能输入单个未知glTF。我们只能借鉴其表征和融合结构，不能使用Clean—Attacked差值作为学生模型输入。

### 2.2 第二优先级：Graphics-LPIPS与YN2023（ACM TOG 2023）

**论文**：Textured Mesh Quality Assessment: Large-scale Dataset and Deep Learning-based Quality Metric  
**论文地址**：<https://doi.org/10.1145/3592786>  
**作者版PDF**：<https://perso.liris.cnrs.fr/guillaume.lavoue/revue/TOG_2023.pdf>  
**数据集**：<https://datasets.liris.cnrs.fr/textured-mesh-quality-assessment-dataset-version1>  
**项目主页**：<https://projet.liris.cnrs.fr/pisco/>  
**相关代码**：<https://github.com/MEPP-team/Graphics-LPIPS>

基本属性：

| 项目 | 内容 |
|---|---|
| 对象 | 带纹理贴图的Mesh |
| 参考类型 | **全参考、投影式** |
| 输入 | Clean与退化Mesh的对应渲染Patch |
| 输出 | 整体质量分数及局部Patch距离 |
| 数据规模 | 55个源模型，343,750个退化刺激 |
| 人工真值 | 3,000个刺激有真实MOS |
| 机器真值 | 其余约340,750个刺激使用Graphics-LPIPS生成Pseudo-MOS |

作者对3,000个代表性样本开展大规模众包实验，4,513名参与者累计给出约14.9万次质量判断。评分采用1～5档：不可察觉、可察觉但不烦扰、轻微烦扰、烦扰、非常烦扰。其余大量样本没有人工MOS，而是由Graphics-LPIPS预测Pseudo-MOS。

与本项目最相关的部分：

- “少量人工MOS＋大量机器Pseudo-MOS”的数据建设路线；
- 使用统一视角和局部图像Patch衡量纹理Mesh感知质量；
- 从庞大的攻击组合中均衡抽取人工评分子集；
- 公开个人评分、MOS、置信区间和Pseudo-MOS，可参考其主观实验组织方式；
- 局部Patch质量经过聚合得到整体质量。

不能直接照搬的部分：Graphics-LPIPS依赖Clean与退化渲染图的成对比较，并且数据以通用物体和压缩退化为主，不是城市摄影测量建筑，也不解决单glTF无参考推理。

### 2.3 第三优先级：SJTU-TMQA

**论文**：SJTU-TMQA: A Quality Assessment Database for Static Mesh with Texture Map  
**论文地址**：<https://arxiv.org/abs/2309.15675>  
**数据库与下载页**：<https://ccccby.github.io/>

基本属性：

| 项目 | 内容 |
|---|---|
| 对象 | 静态纹理Mesh |
| 源模型 | 21个 |
| 退化模型 | 945个 |
| 失真 | 6种单一失真＋2种混合失真 |
| 主观实验 | 73名受试者，实验室环境 |
| 标签 | MOS |

数据包含几何噪声、Mesh简化、纹理JPEG压缩、几何/纹理量化以及复合退化。退化Mesh被渲染成沿固定相机轨迹运动的视频，再由受试者评分。

与本项目最相关的部分：

- 同时包含几何、纹理及混合退化；
- 可用于验证模型是否真正理解几何—纹理交互；
- MOS和模型实体均可下载，适合做外部零样本或跨数据集测试；
- 数据规模适中，作为外部Benchmark比直接混入训练更合适。

局限：主体是通用物体，不是大尺度摄影测量城市建筑；评分基于固定渲染视频，不等于直接检查原生glTF资产。

### 2.4 第四优先级：TSMD

**论文**：TSMD: A Database for Static Color Mesh Quality Assessment Study  
**论文地址**：<https://arxiv.org/abs/2308.01940>  
**腾讯数据页**：<https://multimedia.tencent.com/en/resources/tsmd/>  
**HybridMQA数据接入说明**：<https://github.com/arshafiee/hybridmqa/tree/main/datasets>

基本属性：

| 项目 | 内容 |
|---|---|
| 对象 | 静态彩色Mesh |
| 源模型 | 42个 |
| 退化样本 | 210个 |
| 主要失真 | 有损Mesh压缩 |
| 标签 | 主观MOS |

TSMD主要用于研究静态Mesh压缩对几何和颜色质量的影响。它有真实MOS，但内容数量和攻击多样性都小于YN2023与SJTU-TMQA。

可借鉴：压缩场景的主观实验和率失真评价。局限：攻击集中于编码压缩，不能覆盖摄影测量模型中的孔洞、尖刺、贴图缺失和错位。

### 2.5 第五优先级：VCMesh / CMDM

**数据入口**：<https://github.com/arshafiee/hybridmqa/tree/main/datasets>  
**CMDM说明**：<https://yananehme.github.io/files/Manual_CMDM.pdf>

基本属性：

| 项目 | 内容 |
|---|---|
| 对象 | 顶点着色Mesh，不是UV纹理贴图Mesh |
| 源模型 | 5个 |
| 退化样本 | 480个 |
| 退化 | 几何或顶点颜色失真 |
| 标签 | MOS |

CMDM是面向彩色Mesh的全参考客观指标，融合几何与颜色特征。其直接价值是提供几何/颜色联合质量评价思路和一个人工MOS外部测试集。

局限：只有5个源内容，且颜色存储方式是vertex color，与本项目glTF的UV贴图、材质及多纹理依赖不同。

### 2.6 第六优先级：GMS-3DQA（ACM TOMM 2023）

**论文**：GMS-3DQA: Projection-based Grid Mini-patch Sampling for 3D Model Quality Assessment  
**论文地址**：<https://arxiv.org/abs/2306.05658>  
**官方代码**：<https://github.com/zzc-1998/GMS-3DQA>

基本属性：

| 项目 | 内容 |
|---|---|
| 参考类型 | **无参考** |
| 方法 | 六视图投影＋Grid Mini-patch采样＋Swin Transformer |
| 公开实验数据 | SJTU-PCQA、WPC、WPC2.0 |
| 实际主要对象 | 彩色点云投影 |

它将六个正交投影视图切分并采样为紧凑的Quality Mini-patch Map，再由Swin Transformer预测整体分数。它解决了多视图方法计算量大的问题。

与本项目相关：无参考设定、六视图、局部Patch采样和Transformer质量特征。局限：官方代码和数据流程主要面向点云投影，没有利用Mesh三角面拓扑、UV和材质结构，不能称为现成的glTF Mesh质量评价方案。

### 2.7 第七优先级：NR-3DQA（IEEE TCSVT 2022）

**论文**：No-Reference Quality Assessment for 3D Colored Point Cloud and Mesh Models  
**论文地址**：<https://arxiv.org/abs/2107.02041>  
**期刊页面**：<https://ieeexplore.ieee.org/document/9810024>  
**官方代码**：<https://github.com/zzc-1998/NR-3DQA>

论文研究彩色点云和Mesh的无参考质量评价，使用几何与颜色自然统计特征进行质量回归。它的重要意义是证明无参考3D质量评价可以不输入Clean参考。

局限：公开仓库明确是point cloud version，主要公开实验对应SJTU-PCQA和WPC点云数据库。它适合做传统统计基线或思想对照，不是可直接部署到glTF Mesh的完整实现。

### 2.8 补充工作：FMQM（2025）

**论文**：Textured Mesh Quality Assessment using Geometry and Color Field Similarity  
**论文地址**：<https://arxiv.org/abs/2505.10824>  
**代码**：<https://github.com/yyyykf/FMQM>

FMQM比较参考与退化纹理Mesh的几何场和颜色场相似性，属于全参考方法。它可作为离线机器真值或外部全参考基线，但不满足本项目的单模型无参考部署要求。

## 3. 公开数据集汇总

| 数据集 | 表示 | 源内容 | 退化样本 | 人工标签 | 主要退化 | 与本项目相关性 |
|---|---|---:|---:|---|---|---|
| YN2023/LIRIS | UV纹理Mesh | 55 | 343,750 | 3,000真实MOS，其余Pseudo-MOS | 几何、纹理映射、纹理图像及组合压缩 | 最高 |
| SJTU-TMQA | UV纹理Mesh | 21 | 945 | 全部有MOS | 几何、纹理、量化及混合退化 | 很高 |
| TSMD | 静态彩色Mesh | 42 | 210 | MOS | Mesh压缩 | 中高 |
| VCMesh/CMDM | 顶点着色Mesh | 5 | 480 | MOS | 几何和颜色失真 | 中等 |
| SJTU-PCQA | 彩色点云 | 9 | 378 | MOS | 点云几何/颜色失真 | 间接 |
| WPC | 彩色点云 | 20 | 740 | MOS | 点云压缩与降质 | 间接 |

点云数据库可用于验证投影式模块或迁移学习，但不能替代Mesh数据库，因为它们没有三角面拓扑、UV、材质和纹理依赖。

## 4. 对UrbanPhotoMeshQA的直接借鉴建议

### 4.1 模型结构

按优先级建议：

1. 借鉴HybridMQA的Face图学习、空间投影对齐与Geometry–Texture Cross-Attention；
2. 借鉴GMS-3DQA的六视图Mini-patch采样，降低纹理分支开销；
3. 保持Mesh Face→Patch→Building的层次结构，而不是把Mesh简单转为点云；
4. 使用MAE、PLCC/SRCC相关损失和成对排序损失，但只在Train训练、Val选模；
5. 局部输出应从“图像Patch”进一步绑定到三角面和UV区域。

### 4.2 人工MOS建设

YN2023说明没有必要人工标注全部大样本。对本项目更可行的流程是：

```text
6289个正式样本
→ 分攻击、强度、图幅和建筑形态均衡抽取人工子集
→ 多人进行1～5档匿名主观评分
→ 获得真实MOS
→ 用真实MOS训练或校准无参考模型
→ 为未人工标注样本生成Pseudo-MOS
```

建议先做300条单人Pilot检查评分流程，正式论文阶段再选择约1,000～2,000条组织至少5名评分人。必须保留一部分人工MOS作为完全锁定测试集，不能全部用于训练。

### 4.3 外部数据集的使用方式

目前不建议直接把公开数据集混入主训练集，因为通用物体与香港摄影测量建筑存在明显域差异，而且OBJ/vertex-color与glTF/UV-texture的数据结构不同。更合理的是：

1. 先在本项目数据上训练；
2. 在SJTU-TMQA或YN2023的人工MOS子集上做零样本外部测试；
3. 必要时报告“未微调”和“仅用公开Train微调”两组结果；
4. 不使用公开测试集反向调整主模型；
5. 将HybridMQA、Graphics-LPIPS、GMS-3DQA等作为外部基线或结构消融。

## 5. 本项目相对已有工作的差异化定位

UrbanPhotoMeshQA不是简单复现上述工作，拟形成以下组合创新：

| 维度 | 现有代表工作 | UrbanPhotoMeshQA |
|---|---|---|
| 场景 | 通用物体、压缩模型或点云 | 香港城市摄影测量单体建筑 |
| 格式 | OBJ、顶点颜色、点云PLY | 原生glTF＋BIN＋纹理资产包 |
| 推理参考 | HybridMQA/Graphics-LPIPS需要Clean | 部署时只输入单个待测glTF |
| 退化 | 以编码压缩为主 | 孔洞、QEM、尖刺、细节损失、区域缺失、纹理错位 |
| 三维结构 | 投影或全局评分居多 | Point＋Face Graph＋Morphology＋Texture |
| 局部质量 | 多为二维图像Patch | Mesh Face↔UV↔纹理区域的16个三维Patch |
| 输出 | 通常单一质量分数 | OQI、Geometry、Texture、退化类型、强度和局部Patch |
| 泛化 | 按通用物体划分 | 建筑隔离＋跨图幅Blind评测 |

最有潜力的论文主线是：

> 面向真实城市摄影测量glTF的单模型无参考质量评价，通过拓扑感知Mesh层次表征、几何—纹理对齐融合和跨图幅泛化协议，同时预测全局质量与局部几何/纹理缺陷。

## 6. 结论

- 真正与本项目最接近的方法是HybridMQA，但它是全参考方法；
- 最值得参考的数据建设工作是YN2023，其3,000个真实MOS＋约34万个Pseudo-MOS的路线与我们高度契合；
- 最合适的外部人工MOS Benchmark是SJTU-TMQA和YN2023的3,000条人工子集；
- GMS-3DQA和NR-3DQA是重要无参考方法，但公开实现主要面向点云；
- 目前仍缺少“城市摄影测量单体glTF＋几何/纹理真实文件级退化＋无参考全局/局部联合预测＋跨图幅Blind”的公开完整方案，这正是UrbanPhotoMeshQA最清晰的差异化空间。
