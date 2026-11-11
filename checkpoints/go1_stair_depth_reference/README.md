# Go1 reference-retention refinement

`model_15200.pt` is a V33 training warm-start, **not a selected deployment
policy**. It passed the full 960-trial stair matrix and 448-trial native-terrain
preservation checks. Narrow forward crossing was 280/288 versus 263/288 for
the verified baseline; hard 18 cm stop/resume was 18/24 versus 7/24. It failed
120-second normal-slope survival (11/16 versus 14/16). Both stair families
survived and crossed 16/16 in that long test. The second-seed test was not run.
All seven V33 candidates failed an original acceptance gate; see the summary.

`model_14500.pt` is the previously verified policy, used as a frozen action
reference during training. The new opt-in task adds this loss to PPO:

```text
loss = PPO_loss + 5 * mean_replay(mean_joint((actor_mean - reference_mean)^2))
```

Replay samples are the five non-stair terrain types, the six original
27/30/33 cm stair columns, and the two native-platform stair columns. The loss
does not constrain the eight narrow/adaptive stair columns to the old policy. It does not replace any torque, acceleration, edge, or tracking
reward. It clips the combined actor gradient using the existing norm limit.
The reference network is frozen, including its normalizer; it is excluded
from the optimizer. This is a retention hypothesis, not demonstrated success.

A training-only `reference_mask` observation selects replay rows. It is
included in neither actor nor critic input groups. Actor 45 + depth 1x45x80,
critic 263, actions, robot parameters, rewards and physics remain unchanged.
The exported actor does not need the mask or reference network. Play uses
ordinary manual commands. Original acceptance gates remain unchanged.

`env.yaml` and `agent.yaml` are the warm-start's actual saved configuration;
`reference_env.yaml` and `reference_agent.yaml` describe verified 14500.
`lineage.json` records ancestry: 14500 -> V30b15000 -> V31 15100 -> V33 15200,
703 new updates after 14500. `manifest.json` records data hashes and the new
recipe commit. Keep both checkpoints available when starting this task.

After synchronizing the locally tested commit to the remote server, choose
the lowest-utilization GPU with sufficient free memory, and launch in tmux:

```sh
GO1_PARENT_RUN=logs/rsl_rl/go1_perceptive_velocity/2026-11-11_12-36-03_stair_depth_balanced_v33_remote
mkdir -p "$GO1_PARENT_RUN/params"
cp checkpoints/go1_stair_depth_reference/model_15200.pt "$GO1_PARENT_RUN/"
cp checkpoints/go1_stair_depth_reference/env.yaml \
   checkpoints/go1_stair_depth_reference/agent.yaml "$GO1_PARENT_RUN/params/"
uv run train Mjlab-Velocity-Stair-Depth-Reference-Unitree-Go1-Perceptive \
  --agent.resume True \
  --agent.load-run 2026-11-11_12-36-03_stair_depth_balanced_v33_remote \
  --agent.load-checkpoint model_15200.pt \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 1200 \
  --agent.run-name stair_depth_reference_v34 \
  --agent.logger tensorboard \
  --agent.upload-model False
```

The fixed learning rate is 2.5e-6; checkpoints are saved every 100 iterations.
Predeclared candidates 15300/15500/15700/15900/16100/16300/final16399 correspond
to 101/301/501/701/901/1101/1200 new updates. Verify actual counters, parent
and reference hashes, not filenames alone. Required validation includes the
original full-depth, native-terrain, continuous and second-seed matrices.

V35 restart recipe is recorded in `restart_recipe.json`. It starts directly from the verified14500 checkpoint already stored here, uses its `reference_env.yaml` and `reference_agent.yaml` as parent configuration, and raises the reference coefficient to25 for a finite3000-update experiment. All seven V34 candidates failed original acceptance gates; none is selected or a V35 parent. Rewards, motor constraints and original behavioral acceptance gates remain unchanged.
