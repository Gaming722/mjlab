"""Forward commands with an explicit mid-flight pause and same-speed restart."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.terrains import TerrainEntity

__all__ = ["StopResumeVelocityCommand", "StopResumeVelocityCommandCfg"]


class StopResumeVelocityCommand(UniformVelocityCommand):
  def __init__(self, cfg: StopResumeVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._cycle_cfg = cfg
    self.is_stop_resume_env = torch.zeros_like(self.is_forward_env)
    self.is_paused = torch.zeros_like(self.is_forward_env)
    self.resume_command_b = torch.zeros_like(self.vel_command_b)
    self.pause_begin_time_left = torch.zeros_like(self.time_left)
    self.pause_end_time_left = torch.zeros_like(self.time_left)
    self._cycle_terrain: TerrainEntity | None = None
    self._cycle_columns: torch.Tensor | None = None
    if cfg.terrain_names is not None:
      terrain = env.scene.terrain
      generator = terrain.cfg.terrain_generator if terrain is not None else None
      if generator is None or not generator.curriculum:
        raise ValueError("Terrain-filtered cycles require a curriculum terrain grid.")
      names = list(generator.sub_terrains)
      missing = set(cfg.terrain_names) - set(names)
      if missing:
        raise ValueError(f"Unknown stop/resume terrain names: {sorted(missing)}")
      # Curriculum grids have exactly one column per named sub-terrain.
      self._cycle_terrain = terrain
      self._cycle_columns = torch.tensor(
        [names.index(name) for name in cfg.terrain_names], device=self.device
      )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.is_stop_resume_env[env_ids] = False
    self.is_paused[env_ids] = False
    if self._cycle_cfg.forward_stop_fraction == 0.0:
      return
    eligible = self.is_forward_env[env_ids] & ~self.is_standing_env[env_ids]
    if self._cycle_terrain is not None:
      assert self._cycle_columns is not None
      eligible &= torch.isin(
        self._cycle_terrain.terrain_types[env_ids], self._cycle_columns
      )
    selected = eligible & (
      torch.rand(len(env_ids), device=self.device)
      < self._cycle_cfg.forward_stop_fraction
    )
    ids = env_ids[selected]
    if len(ids) == 0:
      return
    self.is_stop_resume_env[ids] = True
    values = torch.empty(len(ids), device=self.device)
    self.resume_command_b[ids, 0] = values.uniform_(*self._cycle_cfg.resume_speed_range)
    self.resume_command_b[ids, 1:] = 0.0
    before = values.uniform_(*self._cycle_cfg.approach_duration).clone()
    pause = values.uniform_(*self._cycle_cfg.pause_duration).clone()
    after = values.uniform_(*self._cycle_cfg.resume_duration)
    self.pause_begin_time_left[ids] = pause + after
    self.pause_end_time_left[ids] = after
    self.time_left[ids] = before + pause + after
    self.vel_command_b[ids] = self.resume_command_b[ids]

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    super()._update_command(env_ids)
    ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    ids = ids[self.is_stop_resume_env[ids] & ~self.is_standing_env[ids]]
    paused = (self.time_left[ids] <= self.pause_begin_time_left[ids]) & (
      self.time_left[ids] > self.pause_end_time_left[ids]
    )
    self.is_paused[ids] = paused
    self.vel_command_b[ids] = torch.where(
      paused.unsqueeze(-1), 0.0, self.resume_command_b[ids]
    )


@dataclass(kw_only=True)
class StopResumeVelocityCommandCfg(UniformVelocityCommandCfg):
  forward_stop_fraction: float = 0.5
  """Fraction of non-standing forward samples assigned a stop/resume cycle.
  Pure turns, heading/world commands, and standing commands keep their priority.
  Zero retains ordinary uniform sampling without extra random draws."""
  terrain_names: tuple[str, ...] | None = None
  """Restrict cycles to named curriculum-grid terrain columns. Other columns
  retain ordinary commands and resampling durations. None enables all terrains.
  Eligibility is checked again at each resample to follow terrain reassignment."""
  approach_duration: tuple[float, float] = (3.0, 6.0)
  pause_duration: tuple[float, float] = (2.0, 4.0)
  resume_duration: tuple[float, float] = (12.0, 18.0)
  resume_speed_range: tuple[float, float] = (0.3, 0.35)
  """Forward speed held before and after the pause, in m/s."""

  def __post_init__(self):
    super().__post_init__()
    if self.terrain_names is not None and (
      not self.terrain_names
      or any(not name for name in self.terrain_names)
      or len(set(self.terrain_names)) != len(self.terrain_names)
    ):
      raise ValueError("terrain_names must contain distinct, nonempty names.")
    if not np.isfinite(self.forward_stop_fraction) or not (
      0.0 <= self.forward_stop_fraction <= 1.0
    ):
      raise ValueError("forward_stop_fraction must be finite and in [0, 1].")
    for name in ("approach_duration", "pause_duration", "resume_duration"):
      low, high = getattr(self, name)
      if not np.isfinite((low, high)).all() or low <= 0 or high < low:
        raise ValueError(f"{name} must be a positive finite ordered range.")
    low, high = self.resume_speed_range
    if (
      not np.isfinite((low, high)).all()
      or low < self.forward_min_velocity
      or low <= 0
      or high < low
      or high > self.ranges.lin_vel_x[1]
    ):
      raise ValueError("resume_speed_range must fit the forward command limits.")

  def build(self, env: ManagerBasedRlEnv) -> StopResumeVelocityCommand:
    return StopResumeVelocityCommand(self, env)
