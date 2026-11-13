"""Sampling changes survive resume without altering deterministic action means."""

from dataclasses import replace

import torch
from rsl_rl.modules.distribution import GaussianDistribution

from mjlab.tasks.velocity.config.go1.rl_cfg import (
  unitree_go1_stair_depth_exploration_ppo_runner_cfg,
  unitree_go1_stair_depth_restart_ppo_runner_cfg,
)


def test_exploration_runner_changes_only_sampling_floor():
  base = unitree_go1_stair_depth_restart_ppo_runner_cfg()
  cfg = unitree_go1_stair_depth_exploration_ppo_runner_cfg()
  assert cfg.actor.distribution_cfg is not None
  assert cfg.actor.distribution_cfg["std_range"] == (0.55, 1e6)
  assert replace(cfg.actor, distribution_cfg=base.actor.distribution_cfg) == base.actor
  assert replace(cfg, actor=base.actor) == base


def test_floor_applies_after_loading_low_std_and_leaves_mean_export_unchanged():
  parent = GaussianDistribution(12, init_std=0.25)
  resumed = GaussianDistribution(12, init_std=1.0, std_range=(0.55, 1e6))
  resumed.load_state_dict(parent.state_dict(), strict=True)
  means = torch.linspace(-1.0, 1.0, 48).reshape(4, 12)
  parent.update(means)
  resumed.update(means)
  torch.testing.assert_close(resumed.std, torch.full_like(means, 0.55))
  torch.testing.assert_close(parent.std, torch.full_like(means, 0.25))
  torch.testing.assert_close(
    resumed.deterministic_output(means), parent.deterministic_output(means)
  )
  torch.testing.assert_close(resumed.as_deterministic_output_module()(means), means)
  torch.testing.assert_close(
    resumed.state_dict()["std_param"], parent.state_dict()["std_param"]
  )
