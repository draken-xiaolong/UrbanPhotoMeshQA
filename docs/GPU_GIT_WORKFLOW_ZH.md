# GPU 服务器 Git 协作流程

远端仓库：`https://github.com/draken-xiaolong/UrbanPhotoMeshQA.git`。

## 首次部署

```bash
git clone git@github.com:draken-xiaolong/UrbanPhotoMeshQA.git
cd UrbanPhotoMeshQA
git lfs install
git lfs pull
python -m pip install -e '.[train,mesh,vision,test]'
```

仓库保存代码、文档、Manifest、小型实验结果、Checkpoint 和 NPZ 特征。原始 glTF、
攻击资产包以及 12GB 原始缓存继续放在服务器本地数据盘，不复制进 Git。

服务器目录应保持以下映射：

```text
<repo>/                         # Git 工作区
<data-root>/HK3D-Individualised
<data-root>/HK3D-Individualised-Attack
<data-root>/Feature-Cache/Full
```

## 每次实验前

```bash
git status --short
git pull --ff-only origin main
git lfs pull
```

禁止在存在未提交同名修改时直接 pull。正式训练必须显式传入：

```text
--dataset-manifest artifacts/manifests/quality_dataset_formal_seed2026.json
--require-formal
```

候选模型训练阶段不得使用 `--evaluate-locked`。只有 Val 选出的唯一胜者才能运行
Test/Blind 校准与最终评测。

## 上传服务器结果

结果只能写入新的 `artifacts/quality/final/` 或 `artifacts/quality/ablations/` 子目录，
不得覆盖旧发布包。确认输出完整后：

```bash
git status --short
git add <new-result-directory> <changed-code-or-config>
git diff --cached --check
git commit -m "Add <experiment-id> results"
git push origin main
```

`.pt`、`.pth`、`.npz`、`.npy`、`.glb` 和 `.bin` 由 Git LFS 管理。若服务器没有
Git LFS，必须先安装，不能把 LFS pointer 文件当成真实模型或特征文件使用。

## 本地接收服务器结果

```bash
git status --short
git pull --ff-only origin main
git lfs pull
```

拉取后应运行 `pytest -q`，并对发布目录运行 `shasum -a 256 -c SHA256SUMS.txt`。
