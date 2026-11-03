"""Evaluate Go1 stair passage independently of terrain curriculum levels.

Example:
  uv run python -m mjlab.scripts.evaluate_go1_stairs --checkpoints PATH/model.pt
"""

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import RayCastSensorCfg, TerrainHeightSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.terrains.primitive_terrains import BoxPyramidStairsTerrainCfg
from mjlab.utils.torch import configure_torch_backends

TASK = "Mjlab-Velocity-Stairs-Unitree-Go1-Perceptive"


@dataclass
class EvaluationConfig:
  checkpoints: tuple[Path, ...]
  output: Path = Path("artifacts/go1_stairs_evaluation.json")
  heights: tuple[float, ...] = (0.10, 0.14, 0.18)
  speeds: tuple[float, ...] = (0.3, 0.5, 0.8, 1.0)
  directions: tuple[Literal["up", "down"], ...] = ("up", "down")
  trials: int = 16
  duration: float = 25.0
  seed: int = 42
  device: str = "cuda:0"


def passage_masks(
  position: torch.Tensor,
  displacement: torch.Tensor,
  gravity_z: torch.Tensor,
  signed_height: torch.Tensor,
  failed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Require passage across the steps with the expected vertical displacement.

  The evaluated tiles are 8 m wide, with a 1 m border and 3 m center platform.
  At x=3.4 m the robot has passed all steps. A robot that previously fell or
  left the tile sideways cannot subsequently earn a successful passage.
  """
  failed = failed | (gravity_z > -math.cos(math.radians(70)))
  failed |= (position[:, 1].abs() > 3.8) | (position[:, 0] < -3.8)
  height_ok = displacement[:, 2] * signed_height.sign() >= signed_height.abs()
  passed = (position[:, 0] >= 3.4) & height_ok & (gravity_z < -0.5) & ~failed
  return passed, failed


def evaluate(args: EvaluationConfig) -> None:
  if args.trials < 1 or args.duration <= 0:
    raise ValueError("trials and duration must be positive")
  if not args.heights or not args.speeds or not args.directions:
    raise ValueError("heights, speeds and directions must not be empty")
  if any(h <= 0 or h > 0.18 for h in args.heights):
    raise ValueError("heights must be in (0, 0.18] meters")
  if any(v <= 0 for v in args.speeds):
    raise ValueError("speeds must be positive")
  configure_torch_backends()
  torch.set_num_threads(4)
  cfg = load_env_cfg(TASK, play=True)
  cfg.seed = args.seed
  cfg.scene.num_envs = len(args.heights) * len(args.directions) * args.trials
  cfg.terminations = {}
  cfg.events.pop("randomize_terrain", None)
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.15, 0.15),
    "y": (-0.15, 0.15),
    "z": (0.02, 0.02),
    "yaw": (-0.1, 0.1),
  }
  assert cfg.scene.terrain is not None
  terrain = cfg.scene.terrain.terrain_generator
  assert terrain is not None
  labels = []
  sub_terrains = {}
  for direction in args.directions:
    name = "pyramid_stairs_inv" if direction == "up" else "pyramid_stairs"
    for height in args.heights:
      stair = deepcopy(terrain.sub_terrains[name])
      assert isinstance(stair, BoxPyramidStairsTerrainCfg)
      stair.step_height_range = (height, height)
      stair.proportion = 1.0
      sub_terrains[f"{direction}_{height}"] = stair
      labels.append((direction, height))
  terrain.sub_terrains = sub_terrains
  terrain.num_rows = 1
  terrain.num_cols = len(labels)
  terrain.curriculum = True
  command = cfg.commands["twist"]
  # This task uses UniformVelocityCommandCfg; keep every evaluation command fixed.
  assert isinstance(command, UniformVelocityCommandCfg)
  command.heading_command = False
  command.ranges.heading = None
  command.rel_heading_envs = 0.0
  command.rel_world_envs = 0.0
  command.rel_forward_envs = 0.0
  command.rel_standing_envs = 0.0
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (0.0, 0.0)
  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, (RayCastSensorCfg, TerrainHeightSensorCfg)):
      sensor.debug_vis = False
  base = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env = RslRlVecEnvWrapper(base)
  command_term = base.command_manager.get_term("twist")
  assert isinstance(command_term, UniformVelocityCommand)
  command = command_term.cfg
  agent = load_rl_cfg(TASK)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
    env, asdict(agent), device=args.device
  )
  robot = base.scene["robot"]
  terrain_entity = base.scene.terrain
  assert terrain_entity is not None
  types = terrain_entity.terrain_types
  signed_height = torch.tensor(
    [5 * h * (1 if d == "up" else -1) for d, h in labels], device=args.device
  )[types]
  results = []
  try:
    for checkpoint in args.checkpoints:
      runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
      )
      policy = runner.get_inference_policy(device=args.device)
      for speed in args.speeds:
        # Use identical reset noise for every checkpoint and speed.
        base.seed(args.seed)
        command.ranges.lin_vel_x = (speed, speed)
        obs, _ = env.reset()
        start = robot.data.root_link_pos_w.clone()
        passed = torch.zeros(env.num_envs, dtype=torch.bool, device=args.device)
        failed = torch.zeros_like(passed)
        passage_time = torch.full((env.num_envs,), args.duration, device=args.device)
        stop_time = torch.zeros(env.num_envs, device=args.device)
        active_time = torch.zeros_like(stop_time)
        forward_distance = torch.zeros_like(stop_time)
        trunk_contact_time = torch.zeros_like(stop_time)
        lateral_peak = torch.zeros_like(stop_time)
        with torch.no_grad():
          for step in range(math.ceil(args.duration / base.step_dt)):
            active = ~passed & ~failed
            obs, _, _, _ = env.step(policy(obs))
            active_time += active * base.step_dt
            forward_distance += (
              robot.data.root_link_lin_vel_b[:, 0] * active * base.step_dt
            )
            trunk_contacts = base.scene["trunk_ground_touch"].data.found
            assert trunk_contacts is not None
            trunk_touch = (trunk_contacts > 0).flatten(start_dim=1).any(dim=1)
            trunk_contact_time += trunk_touch * active * base.step_dt
            position = robot.data.root_link_pos_w - base.scene.env_origins
            displacement = robot.data.root_link_pos_w - start
            lateral_peak = torch.maximum(
              lateral_peak, displacement[:, 1].abs() * active
            )
            new_pass, new_fail = passage_masks(
              position,
              displacement,
              robot.data.projected_gravity_b[:, 2],
              signed_height,
              failed,
            )
            elapsed = (step + 1) * base.step_dt
            passage_time[new_pass & ~passed] = elapsed
            # Passage is provisional: a subsequent fall still fails the trial.
            failed |= new_fail
            passed |= new_pass
            stopped = robot.data.root_link_lin_vel_b[:, 0].abs() < 0.1
            stop_time += (stopped & ~passed & ~failed) * base.step_dt
        for i, (direction, height) in enumerate(labels):
          mask = types == i
          success = passed[mask] & ~failed[mask]
          times = passage_time[mask][success]
          row = {
            "checkpoint": str(checkpoint.resolve()),
            "direction": direction,
            "height": height,
            "speed": speed,
            "trials": int(mask.sum()),
            "successes": int(success.sum()),
            "failures": int(failed[mask].sum()),
            "success_rate": float(success.float().mean()),
            "mean_passage_s": float(times.mean()) if times.numel() else None,
            "mean_stop_s": float(stop_time[mask].mean()),
            "mean_forward_speed_m_s": float(
              (
                forward_distance[mask] / active_time[mask].clamp_min(base.step_dt)
              ).mean()
            ),
            "mean_trunk_contact_fraction": float(
              (
                trunk_contact_time[mask] / active_time[mask].clamp_min(base.step_dt)
              ).mean()
            ),
            "mean_max_lateral_drift_m": float(lateral_peak[mask].mean()),
            "worst_lateral_drift_m": float(lateral_peak[mask].max()),
          }
          results.append(row)
          print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
          json.dumps(
            {
              "config": {
                **asdict(args),
                "checkpoints": [str(p) for p in args.checkpoints],
                "output": str(args.output),
              },
              "results": results,
            },
            indent=2,
          )
          + "\n"
        )
  finally:
    env.close()


if __name__ == "__main__":
  evaluate(tyro.cli(EvaluationConfig))
