"""Fixed replay covers narrow stair levels without moving reference columns."""

from copy import deepcopy
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.config.go1 import env_cfgs
from mjlab.tasks.velocity.config.go1.env_cfgs import (
  unitree_go1_stair_depth_progress_env_cfg,
  unitree_go1_stair_depth_stratified_env_cfg,
)
from mjlab.tasks.velocity.mdp.terrain_sampling import stratify_narrow_terrain_levels


def test_retention_task_only_adds_native_and_slope_startup_replay(monkeypatch):
  base = unitree_go1_stair_depth_stratified_env_cfg()
  play_base = unitree_go1_stair_depth_stratified_env_cfg(play=True)
  monkeypatch.setattr(
    env_cfgs,
    "unitree_go1_stair_depth_stratified_env_cfg",
    lambda play=False: deepcopy(play_base if play else base),
  )
  cfg = env_cfgs.unitree_go1_stair_depth_retention_env_cfg()
  event = cfg.events.pop("stratify_retention_levels")
  assert event.mode == "startup"
  assert event.func is stratify_narrow_terrain_levels
  assert event.params["terrain_names"] == (
    "pyramid_stairs_native",
    "pyramid_stairs_inv_native",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
  )
  assert cfg == base
  assert env_cfgs.unitree_go1_stair_depth_retention_env_cfg(play=True) == play_base


def test_narrow_levels_cover_full_range_and_emphasize_hard_rows():
  types = torch.tensor([0] * 64 + [1] * 64 + [2] * 3)
  levels = torch.full_like(types, 5)
  origins = torch.arange(16 * 3 * 3).reshape(16, 3, 3).float()
  positions = origins[levels, types].clone()
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(
        curriculum=True, sub_terrains=dict(old=None, narrow=None, tiny=None)
      )
    ),
    terrain_origins=origins,
    terrain_types=types,
    terrain_levels=levels,
    env_origins=positions,
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene=SimpleNamespace(terrain=terrain), cfg=SimpleNamespace(curriculum={})
    ),
  )
  old_positions = positions[:64].clone()
  stratify_narrow_terrain_levels(env, None, ("narrow", "tiny"))
  torch.testing.assert_close(levels[:64], torch.full((64,), 5))
  torch.testing.assert_close(positions[:64], old_positions)
  selected = levels[64:128]
  assert set(selected.tolist()) == set(range(16))
  assert (selected >= 12).sum() == 40
  assert levels[128:].tolist() == [0, 7, 15]
  torch.testing.assert_close(positions[64:], origins[levels[64:], types[64:]])
  first = levels.clone()
  stratify_narrow_terrain_levels(env, None, ("narrow", "tiny"))
  torch.testing.assert_close(levels, first)


def test_stratified_task_changes_only_training_startup_allocation(
  monkeypatch: pytest.MonkeyPatch,
):
  base = unitree_go1_stair_depth_progress_env_cfg()
  play_base = unitree_go1_stair_depth_progress_env_cfg(play=True)
  # Factories contain fresh spec lambdas, so compare copies of the same base.
  monkeypatch.setattr(
    env_cfgs,
    "unitree_go1_stair_depth_progress_env_cfg",
    lambda play=False: deepcopy(play_base if play else base),
  )
  cfg = unitree_go1_stair_depth_stratified_env_cfg()
  event = cfg.events.pop("stratify_narrow_levels")
  assert event.mode == "startup"
  assert len(event.params["terrain_names"]) == 8
  assert cfg == base
  assert unitree_go1_stair_depth_stratified_env_cfg(play=True) == (play_base)
