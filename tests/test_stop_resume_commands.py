"""Stop/resume cycles retain the sampled speed and respect reset boundaries."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.stop_resume_command import StopResumeVelocityCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg


def _env(num_envs=8):
  data = SimpleNamespace(
    heading_w=torch.zeros(num_envs),
    root_link_pos_w=torch.zeros(num_envs, 3),
    root_link_lin_vel_b=torch.zeros(num_envs, 3),
    root_link_ang_vel_b=torch.zeros(num_envs, 3),
  )
  return cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={"robot": SimpleNamespace(data=data)},
      num_envs=num_envs,
      device="cpu",
      step_dt=0.02,
      episode_length_buf=torch.ones(num_envs, dtype=torch.long),
    ),
  )


def _cfg():
  return StopResumeVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(4.0, 8.0),
    rel_forward_envs=1.0,
    forward_min_velocity=0.2,
    forward_stop_fraction=1.0,
    approach_duration=(1.0, 1.0),
    pause_duration=(2.0, 2.0),
    resume_duration=(3.0, 3.0),
    resume_speed_range=(0.25, 0.25),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-1.0, 1.0)
    ),
  )


def test_cycle_pauses_then_restores_same_speed_and_resamples():
  term = _cfg().build(_env())
  term.reset(torch.arange(8))
  expected = torch.zeros(8, 3)
  expected[:, 0] = 0.25
  torch.testing.assert_close(term.command, expected)
  term.compute(1.0)
  assert term.is_paused.all()
  assert torch.count_nonzero(term.command) == 0
  term.compute(1.0)
  assert term.is_paused.all()
  term.compute(1.0)
  assert not term.is_paused.any()
  torch.testing.assert_close(term.command, expected)
  term.compute(2.9)
  torch.testing.assert_close(term.command, expected)
  term.compute(0.11)
  assert (term.command_counter == 2).all()
  torch.testing.assert_close(term.command, expected)


def test_partial_reset_and_zero_dt_do_not_advance_other_cycle():
  term = _cfg().build(_env(2))
  term.reset(torch.arange(2))
  term.compute(1.0)
  term.reset(torch.tensor([0]))
  term.compute(0.0, env_ids=torch.tensor([0]))
  torch.testing.assert_close(term.time_left, torch.tensor([6.0, 5.0]))
  torch.testing.assert_close(term.command[:, 0], torch.tensor([0.25, 0.0]))
  term.compute(torch.tensor([0.0, 0.5]))
  torch.testing.assert_close(term.time_left, torch.tensor([6.0, 4.5]))
  torch.testing.assert_close(term.command[:, 0], torch.tensor([0.25, 0.0]))


@pytest.mark.parametrize("standing,turning", [(1.0, 0.0), (0.0, 1.0)])
def test_standing_and_pure_turns_are_never_overridden(standing, turning):
  cfg = replace(_cfg(), rel_standing_envs=standing, rel_turning_envs=turning)
  term = cfg.build(_env())
  term.reset(torch.arange(8))
  term.compute(0.0)
  original = term.command.clone()
  term.compute(1.0)
  assert not term.is_stop_resume_env.any()
  torch.testing.assert_close(term.command, original)
  if standing:
    assert torch.count_nonzero(term.command) == 0
  else:
    assert torch.count_nonzero(term.command[:, :2]) == 0


def test_disabled_cycles_preserve_uniform_rng_and_command_sequence():
  cfg = replace(
    _cfg(), forward_stop_fraction=0.0, rel_forward_envs=0.5, rel_turning_envs=0.25
  )
  base = UniformVelocityCommandCfg(
    **{
      k: v
      for k, v in vars(cfg).items()
      if k in UniformVelocityCommandCfg.__dataclass_fields__
    }
  )
  commands = []
  for command_cfg in (base, cfg):
    torch.manual_seed(37)
    term = command_cfg.build(_env())
    term.reset(torch.arange(8))
    term.compute(0.02)
    commands.append((term.command.clone(), term.time_left.clone()))
  torch.testing.assert_close(commands[0][0], commands[1][0])
  torch.testing.assert_close(commands[0][1], commands[1][1])


@pytest.mark.parametrize(
  "kwargs",
  [
    {"forward_stop_fraction": 1.1},
    {"pause_duration": (0.0, 2.0)},
    {"resume_speed_range": (0.1, 0.3)},
    {"resume_speed_range": (0.2, 0.8)},
  ],
)
def test_invalid_cycle_configuration_fails_early(kwargs):
  with pytest.raises(ValueError):
    replace(_cfg(), **kwargs)


def _terrain_env(curriculum=True):
  env = cast(SimpleNamespace, _env(4))
  scene = MagicMock()
  scene.__getitem__.side_effect = env.scene.__getitem__
  scene.terrain = SimpleNamespace(
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(
        curriculum=curriculum, sub_terrains={"flat": None, "stairs": None}
      )
    ),
    terrain_types=torch.tensor([0, 1, 0, 1]),
  )
  env.scene = scene
  return cast(ManagerBasedRlEnv, env)


def test_terrain_cycles_keep_other_columns_uniform_and_follow_reassignment():
  env = _terrain_env()
  term = replace(_cfg(), terrain_names=("stairs",)).build(env)
  ids = torch.arange(4)
  term.reset(ids)
  assert term.is_stop_resume_env.tolist() == [False, True, False, True]
  assert ((term.time_left[[0, 2]] >= 4) & (term.time_left[[0, 2]] <= 8)).all()
  term.compute(1.0)
  assert term.is_paused.tolist() == [False, True, False, True]
  assert (term.command[[0, 2], 0] > 0).all()
  assert env.scene.terrain is not None
  env.scene.terrain.terrain_types[:] = torch.tensor([1, 0, 1, 0])
  term.reset(ids)
  assert term.is_stop_resume_env.tolist() == [True, False, True, False]


@pytest.mark.parametrize("names", [(), ("",), ("stairs", "stairs")])
def test_invalid_terrain_cycle_names_fail_early(names):
  with pytest.raises(ValueError, match="terrain_names"):
    replace(_cfg(), terrain_names=names)


def test_terrain_cycles_reject_unknown_names_and_random_grids():
  with pytest.raises(ValueError, match="Unknown"):
    replace(_cfg(), terrain_names=("absent",)).build(_terrain_env())
  with pytest.raises(ValueError, match="curriculum"):
    replace(_cfg(), terrain_names=("stairs",)).build(_terrain_env(False))
