"""Forward commands with an explicit mid-flight pause and same-speed restart."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

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

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.is_stop_resume_env[env_ids] = False
    self.is_paused[env_ids] = False
    if self._cycle_cfg.forward_stop_fraction == 0.0:
      return
    eligible = self.is_forward_env[env_ids] & ~self.is_standing_env[env_ids]
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
  approach_duration: tuple[float, float] = (3.0, 6.0)
  pause_duration: tuple[float, float] = (2.0, 4.0)
  resume_duration: tuple[float, float] = (12.0, 18.0)
  resume_speed_range: tuple[float, float] = (0.3, 0.35)
  """Forward speed held before and after the pause, in m/s."""

  def __post_init__(self):
    super().__post_init__()
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
