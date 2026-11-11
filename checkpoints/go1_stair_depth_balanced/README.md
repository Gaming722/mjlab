# Go1 balanced stair-depth warm-start

`model_15100.pt` is a training parent, **not a deployment-selected policy**.
It passed the original 960-trial stair-depth and 448-trial native-terrain
preservation checks, including 20/24 hard 18 cm stop/resume trials. Its
120-second continuous test failed normal-stair survival (13/16 versus 16/16
for the verified baseline). The second-seed test was therefore not run.
See `evaluation_summary.json`; all acceptance gates remain unchanged.

The verified ancestry is `14500 -> V30b15000 -> V31 15100`: 602 new learning
updates after 14500, including 101 after 15000. `lineage.json` preserves the
earlier ancestry. `env.yaml` and `agent.yaml` are the parent's actual saved
configuration. `manifest.json` records hashes and the new recipe commit.

V32's four candidates failed unchanged preservation or hard stop/resume
checks. A command-only coverage diagnostic also found that longer cycles
reduced pure-turn time from approximately 21% to 13%. This suggests a coverage
problem; it does not prove a reward-specific cause of behavioral regression.

The opt-in Balanced task keeps rewards, observations, actions, motor limits,
and acceleration constraints. It rehearses 22--33 cm treads and 2--18 cm
heights, devotes 25% of each stair family to native 30 cm/3 m platforms, and
restricts longer stop/resume cycles to the other stair columns. Other terrain
and native-platform columns retain ordinary command durations. Standing and
pure-turn commands retain priority. Each longest scripted cycle fits within
the 45-second training episode. Play retains manual commands.

After choosing a genuinely idle GPU on the shared server, copy this parent
into the runner's log layout and launch inside tmux:

```sh
GO1_PARENT_RUN=logs/rsl_rl/go1_perceptive_velocity/2026-11-10_20-42-37_stair_depth_rehearsal_v31
mkdir -p "$GO1_PARENT_RUN/params"
cp checkpoints/go1_stair_depth_balanced/model_15100.pt "$GO1_PARENT_RUN/"
cp checkpoints/go1_stair_depth_balanced/env.yaml \
   checkpoints/go1_stair_depth_balanced/agent.yaml "$GO1_PARENT_RUN/params/"
uv run train Mjlab-Velocity-Stair-Depth-Balanced-Unitree-Go1-Perceptive \
  --agent.resume True \
  --agent.load-run 2026-11-10_20-42-37_stair_depth_rehearsal_v31 \
  --agent.load-checkpoint model_15100.pt \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 1200 \
  --agent.run-name stair_depth_balanced_v33 \
  --agent.logger tensorboard \
  --agent.upload-model False
```

The fixed learning rate is 2.5e-6. Saved iterations 15200, 15400, 15600,
15800, 16000 and 16200 correspond to 101, 301, 501, 701, 901 and 1101 new
updates; final 16299 corresponds to 1200. Run names distinguish these weights
from similarly numbered checkpoints in other runs. Verify counters and
parent hashes, not filenames alone. All candidates require the original
full-depth, native preservation, continuous and second-seed evaluations.

With center spawning, `pyramid_stairs` starts on the top platform and its
first outward flight descends; `pyramid_stairs_inv` starts in the pit and
its first outward flight ascends. Maneuvers can traverse both directions.
