# 发布模型训练谱系审计

审计日期：2026-09-03。

`global_quality_head.pt` 和 `local_patch_head.pt` 来源于最初的 3496 条训练流程。
随后质量控制发现三条 QEM heavy 与对应 medium 资产完全相同，并将它们从正式
3493 条评测清单剔除。其中两条位于 Train，一条位于 Test。

因此：

- 本目录报告的 Val/Test/Blind 指标是在剔除重复项后的正式清单上重新计算的；
- 现有 Checkpoint 的训练输入仍包含 Train 中两条后来剔除的重复记录；
- 它应保留为 `release_seed2026_v1` 历史基准，不应声称是严格使用 1518 条正式
  Train 训练得到的 Checkpoint；
- 后续模型必须通过训练入口的 `--dataset-manifest ... --require-formal` 校验，
  并在 Checkpoint 内保存有序样本 SHA-256、Split 数量和正式排除数量。

这一问题不改变本目录已经报告的 3493 条评测数值，但后续候选模型必须以统一的
正式训练清单进行公平比较。
