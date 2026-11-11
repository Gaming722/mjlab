# Go1 楼梯深度微调父模型

这里的 `model_15399.pt` 是下一轮训练的 warm-start，**不是通过全部验收的部署策略**。
它来自 `2026-11-10_20-42-37_stair_depth_rehearsal_v31`，比已验证的
14500 累计多 901 次更新。完整 960 次楼梯测试通过，最难停止再启动为
21/24；原生正金字塔楼梯保留测试为 55/64，基线 57/64，未达到 97% 保留门槛。
连续行走和额外尺寸测试因此未继续运行。

方向按中心出生点解释：`pyramid_stairs` 从顶平台出发，首段向外是下楼；
`pyramid_stairs_inv` 从坑底出发，首段向外是上楼。转向测试整段可能包含
两个方向，不能仅凭生成器名称把整段轨迹当作单向上下楼。

`env.yaml`、`agent.yaml` 是父模型的实际训练配置。新一轮使用代码注册的
LongStop 配方：增加原生 30 cm／3 m 顶平台楼梯回放，停顿覆盖 2–10 秒，
单回合 45 秒，学习率 2.5e-6；奖励、观测、动作与力矩/平滑性约束保持不变。
新配方的能力仍需行为验收，实际显存需求以远程 GPU 预检为准。

`manifest.json` 记录各数据文件的 SHA256；`lineage.json` 保存历史父链，
历史节点仅作溯源，不需要下载所有祖先模型来恢复本父模型。

## 远程服务器继续训练

先在服务器切换到该实验分支，使用已配置好的 GPU 训练环境：

```sh
git fetch origin
git switch feat/go1-stair-depth-finetune
git pull --ff-only
```

将 Git 中的父模型及配置放入训练器识别的日志目录。只复制这三个文件，
不会从 Git 下载其余本地训练日志：

```sh
GO1_PARENT_RUN=logs/rsl_rl/go1_perceptive_velocity/2026-11-10_20-42-37_stair_depth_rehearsal_v31
mkdir -p "$GO1_PARENT_RUN/params"
cp checkpoints/go1_stair_depth_longstop/model_15399.pt "$GO1_PARENT_RUN/"
cp checkpoints/go1_stair_depth_longstop/env.yaml \
   checkpoints/go1_stair_depth_longstop/agent.yaml "$GO1_PARENT_RUN/params/"
```

模型 SHA256 应为：
`6ec5539167f5b07265db8c345ef98d411e6a547973b12b52dd150cdb98f50ae8`。

```sh
uv run train Mjlab-Velocity-Stair-Depth-LongStop-Unitree-Go1-Perceptive \
  --agent.resume True \
  --agent.load-run 2026-11-10_20-42-37_stair_depth_rehearsal_v31 \
  --agent.load-checkpoint model_15399.pt \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 400 \
  --agent.run-name stair_depth_longstop_v32 \
  --agent.logger tensorboard \
  --agent.upload-model False
```

最多新增 400 次更新。由于恢复后的 iteration 从 15399 开始计数，建议检查
15500（约新增 102 次）、15600（202 次）、15700（302 次）及最终
15798（400 次）。新 run 内较早的同名 checkpoint 不等于这里的父文件。
保留新 run 的 `params/` 和 checkpoint，后续按原门槛评估窄踏面、宽楼梯、
其他地形、连续行走及另一组尺寸/种子；不能仅根据训练 terrain level 选模。
