"""Turning commands override forward/heading/world groups, but not standing."""

from types import SimpleNamespace
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def test_turning_preserves_sampled_yaw_and_zeroes_planar_velocity():
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={"robot": SimpleNamespace()},
      num_envs=32,
      device="cpu",
    ),
  )
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(4.0, 8.0),
    rel_forward_envs=1.0,
    rel_turning_envs=1.0,
    rel_world_envs=1.0,
    rel_heading_envs=1.0,
    rel_standing_envs=0.0,
    heading_command=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.3, 0.3),
      lin_vel_y=(0.2, 0.2),
      ang_vel_z=(0.8, 0.8),
      heading=(-1.0, 1.0),
    ),
  )
  term = cfg.build(env)
  term._resample_command(torch.arange(32))
  term._resolve_command(torch.zeros(32))
  torch.testing.assert_close(term.command[:, :2], torch.zeros(32, 2))
  torch.testing.assert_close(term.command[:, 2], torch.full((32,), 0.8))
  assert not term.is_forward_env.any()
  assert not term.is_heading_env.any()
  assert not term.is_world_env.any()
  term.is_standing_env[:] = True
  term._resolve_command(torch.zeros(32))
  assert torch.count_nonzero(term.command) == 0
