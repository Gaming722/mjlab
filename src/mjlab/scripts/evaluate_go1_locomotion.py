"""Evaluate continuous Go1 locomotion, including falls after a terrain crossing.

Unlike stair-passage scoring, failure detection remains active for the full trial.
"""

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import RayCastSensorCfg, TerrainHeightSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainOutput
from mjlab.utils.torch import configure_torch_backends

TASK = "Mjlab-Velocity-Robust-Unitree-Go1-Perceptive"


@dataclass(kw_only=True)
class FixedDifficultyTerrainCfg(SubTerrainCfg):
  terrain: SubTerrainCfg
  difficulty: float

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty
    self.terrain.size = self.size
    return self.terrain.function(self.difficulty, spec, rng)


@dataclass
class EvaluationConfig:
  checkpoints: tuple[Path, ...]
  output: Path = Path("artifacts/go1_locomotion/evaluation.json")
  terrains: tuple[str, ...] = (
    "flat",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "random_rough",
    "wave_terrain",
  )
  difficulties: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
  profiles: tuple[str, ...] = ("slow", "medium", "maneuver", "yaw")
  trials: int = 8
  duration: float = 40.0
  spawn_yaw: float = math.pi
  spawn_xy: float = 0.25
  friction: float | None = None
  seed: int = 43
  device: str = "cuda:0"


def scheduled_command(profile: str, elapsed: float) -> tuple[float, float, float]:
  if profile == "slow":
    return (0.3, 0.0, 0.0)
  if profile == "medium":
    return (0.6, 0.0, 0.0)
  if profile == "maneuver":
    stages = (
      (0.35, 0.0, 0.0),
      (0.25, 0.0, 0.5),
      (0.0, 0.0, 0.0),
      (0.25, 0.0, -0.5),
      (0.25, 0.2, 0.0),
      (-0.25, 0.0, 0.0),
      (0.0, 0.0, 1.0),
      (0.0, 0.0, -1.0),
    )
    return stages[min(int(elapsed / 5), len(stages) - 1)]
  if profile == "yaw":
    stages = (
      (0.35, 0.0, 0.0),
      (0.0, 0.0, 1.0),
      (-0.3, 0.0, 0.0),
      (0.0, 0.0, -1.0),
      (0.0, 0.0, 0.0),
    )
    return stages[min(int(elapsed / 8), len(stages) - 1)]
  raise ValueError(f"Unknown command profile: {profile}")


def update_failure(
  failed: torch.Tensor,
  gravity_z: torch.Tensor,
  trunk_contact_steps: torch.Tensor,
  dt: float,
) -> torch.Tensor:
  """Keep checking failures after crossing; reject sustained belly contact."""
  return (
    failed
    | (gravity_z > -math.cos(math.radians(70)))
    | (trunk_contact_steps * dt >= 0.5)
  )


def evaluate(args: EvaluationConfig) -> None:
  if args.trials < 1 or args.duration <= 0 or not args.terrains:
    raise ValueError("trials, duration, and terrains must be nonempty/positive")
  if not args.difficulties or any(not 0 <= d <= 1 for d in args.difficulties):
    raise ValueError("difficulties must be in [0, 1]")
  for profile in args.profiles:
    scheduled_command(profile, 0.0)
  configure_torch_backends()
  torch.set_num_threads(4)
  cfg = load_env_cfg(TASK, play=True)
  cfg.seed = args.seed
  if args.friction is not None:
    if args.friction <= 0:
      raise ValueError("friction must be positive")
    cfg.events["foot_friction_slide"].params["ranges"] = (args.friction, args.friction)
  cfg.terminations = {}
  cfg.events.pop("randomize_terrain", None)
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-args.spawn_xy, args.spawn_xy),
    "y": (-args.spawn_xy, args.spawn_xy),
    "z": (0.02, 0.02),
    "yaw": (-args.spawn_yaw, args.spawn_yaw),
  }
  assert cfg.scene.terrain is not None
  terrain = cfg.scene.terrain.terrain_generator
  assert terrain is not None
  labels = [(name, d) for name in args.terrains for d in args.difficulties]
  terrain.sub_terrains = {
    f"{name}_{d}": FixedDifficultyTerrainCfg(
      terrain=deepcopy(terrain.sub_terrains[name]), difficulty=d, proportion=1.0
    )
    for name, d in labels
  }
  terrain.num_rows = 1
  terrain.num_cols = len(labels)
  terrain.curriculum = True
  terrain.border_width = 40.0
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.scene.num_envs = len(labels) * args.trials
  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, (RayCastSensorCfg, TerrainHeightSensorCfg)):
      sensor.debug_vis = False
  base = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env = RslRlVecEnvWrapper(base)
  command = base.command_manager.get_term("twist")
  assert isinstance(command, UniformVelocityCommand)
  command.cfg.heading_command = False
  command.cfg.ranges.heading = None
  command.cfg.rel_forward_envs = 0.0
  command.cfg.rel_turning_envs = 0.0
  command.cfg.rel_world_envs = 0.0
  command.cfg.rel_heading_envs = 0.0
  command.cfg.rel_standing_envs = 0.0
  command.cfg.resampling_time_range = (1e9, 1e9)
  command.cfg.ranges.lin_vel_x = (0.0, 0.0)
  command.cfg.ranges.lin_vel_y = (0.0, 0.0)
  command.cfg.ranges.ang_vel_z = (0.0, 0.0)
  agent = load_rl_cfg(TASK)
  runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
    env, asdict(agent), device=args.device
  )
  robot = base.scene["robot"]
  terrain_entity = base.scene.terrain
  assert terrain_entity is not None
  types = terrain_entity.terrain_types
  results = []
  try:
    for checkpoint in args.checkpoints:
      runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=args.device)
      policy = runner.get_inference_policy(device=args.device)
      for profile in args.profiles:
        base.seed(args.seed)
        obs, _ = env.reset()
        failed = torch.zeros(env.num_envs, dtype=torch.bool, device=args.device)
        crossed = torch.zeros_like(failed)
        failure_time = torch.full((env.num_envs,), args.duration, device=args.device)
        contact_steps = torch.zeros(env.num_envs, device=args.device)
        alive_time = torch.zeros_like(contact_steps)
        trunk_time = torch.zeros_like(contact_steps)
        distance = torch.zeros_like(contact_steps)
        stall_time = torch.zeros_like(contact_steps)
        sums = torch.zeros(env.num_envs, 9, device=args.device)
        previous_motion = torch.zeros(env.num_envs, 3, device=args.device)
        previous_heading = robot.data.heading_w.clone()
        low_pass = torch.zeros_like(previous_motion)
        alpha = base.step_dt / (1 / (2 * math.pi * 5) + base.step_dt)
        with torch.no_grad():
          for step in range(math.ceil(args.duration / base.step_dt)):
            elapsed = step * base.step_dt
            target = torch.tensor(
              scheduled_command(profile, elapsed), device=args.device
            )
            command.vel_command_b[:] = target
            # Refresh the command observation before acting, as the Viser joystick does.
            obs = env.get_observations()
            active = ~failed
            obs, _, _, _ = env.step(policy(obs))
            velocity = robot.data.root_link_lin_vel_b
            omega = robot.data.root_link_ang_vel_b
            motion = torch.stack((velocity[:, 2], omega[:, 0], omega[:, 1]), dim=-1)
            if step == 0:
              previous_motion.copy_(motion)
              low_pass.copy_(motion)
            low_pass.lerp_(motion, alpha)
            high_pass = motion - low_pass
            delta = (motion - previous_motion) / base.step_dt
            previous_motion.copy_(motion)
            contacts = base.scene["trunk_ground_touch"].data.found
            assert contacts is not None
            trunk_touch = (contacts > 0).flatten(start_dim=1).any(dim=1)
            contact_steps = torch.where(trunk_touch, contact_steps + 1, 0)
            new_failed = update_failure(
              failed, robot.data.projected_gravity_b[:, 2], contact_steps, base.step_dt
            )
            failure_time[new_failed & ~failed] = (step + 1) * base.step_dt
            failed = new_failed
            position = robot.data.root_link_pos_w - base.scene.env_origins
            crossed |= (position[:, :2].abs().amax(dim=-1) >= 3.4) & ~failed
            alive_time += active * base.step_dt
            distance += velocity[:, :2].norm(dim=-1) * active * base.step_dt
            trunk_time += trunk_touch * active * base.step_dt
            active_command = target[:2].norm() > 0.1
            projected = (velocity[:, :2] * target[:2]).sum(-1) / target[
              :2
            ].norm().clamp_min(1e-6)
            stall_time += (active_command & (projected < 0.1) & active) * base.step_dt
            error_xy = (velocity[:, :2] - target[:2]).square().sum(-1)
            heading_delta = robot.data.heading_w - previous_heading
            heading_rate = (
              torch.atan2(heading_delta.sin(), heading_delta.cos()) / base.step_dt
            )
            previous_heading.copy_(robot.data.heading_w)
            error_yaw = (heading_rate - target[2]).square()
            action_delta = base.action_manager.action - base.action_manager.prev_action
            values = torch.stack(
              (
                motion[:, 0].square(),
                motion[:, 1:].square().sum(-1),
                high_pass[:, 0].square(),
                high_pass[:, 1:].square().sum(-1),
                delta[:, 0].square(),
                delta[:, 1:].square().sum(-1),
                error_xy,
                error_yaw,
                action_delta.square().sum(-1),
              ),
              dim=-1,
            )
            # Ignore the initial settling transient when measuring vibration.
            if elapsed >= 1.0:
              sums += values * active[:, None] * base.step_dt
        for i, (name, difficulty) in enumerate(labels):
          mask = types == i
          count = int(mask.sum())
          valid_time = (alive_time[mask] - 1.0).clamp_min(base.step_dt)
          survived = ~failed[mask]
          # A model that falls early must not appear smooth because it stopped
          # contributing samples. Report motion quality only for full survivors.
          rms = torch.sqrt(sums[mask] / valid_time[:, None])[survived].mean(0)
          metrics = [float(v) if survived.any() else None for v in rms]
          row = {
            "checkpoint": str(checkpoint.resolve()),
            "terrain": name,
            "difficulty": difficulty,
            "profile": profile,
            "trials": count,
            "survivors": int(survived.sum()),
            "survival_rate": float(survived.float().mean()),
            "crossed_and_survived": int((crossed[mask] & survived).sum()),
            "mean_failure_or_end_s": float(failure_time[mask].mean()),
            "mean_distance_m": float(distance[mask].mean()),
            "mean_stall_s": float(stall_time[mask].mean()),
            "mean_trunk_contact_fraction": float(
              (trunk_time[mask] / alive_time[mask].clamp_min(base.step_dt)).mean()
            ),
            "lin_vel_z_rms": metrics[0],
            "ang_vel_xy_rms": metrics[1],
            "lin_vel_z_highpass_rms": metrics[2],
            "ang_vel_xy_highpass_rms": metrics[3],
            "vertical_acc_rms": metrics[4],
            "roll_pitch_acc_rms": metrics[5],
            "velocity_xy_rmse": metrics[6],
            "yaw_rate_rmse": metrics[7],
            "action_rate_rms": metrics[8],
          }
          results.append(row)
          print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        config = {
          **asdict(args),
          "checkpoints": [str(p) for p in args.checkpoints],
          "output": str(args.output),
        }
        args.output.write_text(
          json.dumps({"config": config, "results": results}, indent=2) + "\n"
        )
  finally:
    env.close()


if __name__ == "__main__":
  evaluate(tyro.cli(EvaluationConfig))
