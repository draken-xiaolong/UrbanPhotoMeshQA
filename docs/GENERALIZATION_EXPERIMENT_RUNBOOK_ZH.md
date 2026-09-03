# Blind 泛化最小实验运行手册

本轮目的不是在 Test/Blind 上搜索模型，而是用固定随机种子 `2026`，仅通过 Train/Val
筛选四种冻结 Base 的轻量域泛化策略。只有 Val 选择完成并冻结唯一候选后，才允许对
Test/Blind 做一次最终评测。

## 第一阶段候选

| ID | 归一化 | 采样 | 域鲁棒损失 |
|---|---|---|---|
| B0 | Train mean/std | 样本均匀 | 无 |
| B1 | Train median/IQR | 样本均匀 | 无 |
| B2 | Train median/IQR | Train 图幅均衡 | 无 |
| B3 | Train median/IQR | Train 图幅均衡 | smooth worst-tile |

四组都只训练 `four_branch` Quality Head。部分解冻和端到端训练需要从原始缓存经过
Point/Mesh/Texture Encoder 反向传播，不能使用已经冻结的特征 NPZ 冒充，因此推迟到
第二阶段；只有第一阶段没有达到 Val promotion gate 时才实现。

## GPU 启动

先确保仓库无未提交修改：

```bash
git pull --ff-only origin main
git lfs pull
git status --short
```

先查看将要执行的命令，不启动训练：

```bash
python scripts/run_quality_generalization_minimal.py
```

确认服务器数据目录映射后正式执行：

```bash
python scripts/run_quality_generalization_minimal.py \
  --execute \
  --data-root /root/autodl-tmp/UrbanPhotoMeshQA/data
```

脚本首先执行 fail-fast 审计，包括：CUDA、正式 Manifest 的 3493 条顺序、3 条排除、
特征与真值 NPZ、Point/Mesh Checkpoint、发布模型 SHA-256 和 Git LFS 实体文件。审计
失败时不会开始训练。

中断后续跑：

```bash
python scripts/run_quality_generalization_minimal.py \
  --execute --resume \
  --data-root /root/autodl-tmp/UrbanPhotoMeshQA/data
```

每组输出位于：

```text
artifacts/quality/ablations/generalization_minimal_seed2026_v1/
├── gpu_environment_audit.json
├── B0_formal_frozen/
├── B1_robust_norm/
├── B2_robust_tile_balanced/
├── B3_robust_tile_worst/
└── val_selection.json
```

候选训练过程不会加载 Test/Blind NPZ。汇总脚本还会检查每个 `results.json` 只能包含
Val；一旦发现锁定集指标便拒绝选择。

## Val 选择与解锁条件

主指标为 Val OQI SRCC，MAE 和 PLCC 仅用于打破平局。相对 B0 的 promotion gate：

- OQI SRCC 至少提升 `0.01`；
- Geometry SRCC 下降不超过 `0.02`；
- Texture SRCC 下降不超过 `0.02`。

没有候选通过时，选择仍固定为 B0，而不是查看 Blind 后改选。将
`val_selection.json` 和四组结果提交、推送并人工确认后，才对其中记录的唯一
`selected_checkpoint` 执行一次 Val 校准及 Test/Blind 评测。

## 回传结果

```bash
git add artifacts/quality/ablations/generalization_minimal_seed2026_v1
git diff --cached --check
git commit -m "Add seed-2026 Val generalization screen"
git push origin main
```

不要覆盖 `artifacts/quality/final/release_seed2026_v1`。最终胜者应发布到新的 `final/`
子目录。
