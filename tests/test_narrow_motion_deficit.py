"""Terrain-local progress pressure keeps all existing safety terms intact."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg
from mjlab.tasks.velocity.config.go1.env_cfgs import (
  unitree_go1_stair_depth_progress_env_cfg,
  unitree_go1_stair_depth_restart_env_cfg,
)
from mjlab.tasks.velocity.mdp.rewards import terrain_motion_deficit
from mjlab.tasks.velocity.mdp.stop_resume_command import StopResumeVelocityCommandCfg


class _Scene(dict):
  terrain: SimpleNamespace


def test_scoped_deficit_exempts_pause_turn_and_replay_and_follows_reassignment():
  commands = torch.tensor(
    [
      [0.25, 0.0, 0.0],
      [0.25, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0],
      [0.25, 0.0, 0.0],
      [0.25, 0.0, 0.0],
    ]
  )
  velocity = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.2, 0.0, 0.0],
      [0.0, 0.2, 0.0],
    ]
  )
  scene = _Scene(
    robot=SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=velocity))
  )
  scene.terrain = SimpleNamespace(
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(
        curriculum=True, sub_terrains={"flat": None, "narrow": None}
      )
    ),
    terrain_types=torch.tensor([1, 0, 1, 1, 1, 1]),
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene=scene,
      device="cpu",
      command_manager=SimpleNamespace(get_command=lambda _: commands),
    ),
  )
  params = dict(terrain_names=("narrow",), command_name="twist")
  term = terrain_motion_deficit(
    RewardTermCfg(func=terrain_motion_deficit, weight=-8.0, params=params), env
  )
  torch.testing.assert_close(
    term(env, ("narrow",), "twist"), torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
  )
  scene.terrain.terrain_types[:2] = torch.tensor([0, 1])
  torch.testing.assert_close(
    term(env, ("narrow",), "twist"), torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
  )


@pytest.mark.parametrize("play", [False, True])
def test_progress_task_preserves_all_existing_terms_and_command_contract(play):
  base = unitree_go1_stair_depth_restart_env_cfg(play=play)
  cfg = unitree_go1_stair_depth_progress_env_cfg(play=play)
  for key, term in base.rewards.items():
    assert cfg.rewards[key] == term
  for name in (
    "actions",
    "sim",
    "observations",
    "commands",
    "events",
    "terminations",
    "curriculum",
  ):
    assert getattr(cfg, name) == getattr(base, name)
  if play:
    assert cfg.rewards == base.rewards
  else:
    assert set(cfg.rewards) - set(base.rewards) == {"narrow_motion_deficit"}
    cost = cfg.rewards["narrow_motion_deficit"]
    assert cost.weight == -8.0
    command = cfg.commands["twist"]
    assert isinstance(command, StopResumeVelocityCommandCfg)
    assert cost.params["terrain_names"] == command.terrain_names
