"""Continuous locomotion scoring must not stop checking after a passage."""

import mujoco
import numpy as np
import pytest
import torch

from mjlab.scripts.evaluate_go1_locomotion import (
  FixedDifficultyTerrainCfg,
  scheduled_command,
  update_failure,
)
from mjlab.terrains.config import pyramid_stairs_inv


def test_late_fall_is_latched_even_if_robot_recovers():
  failed = torch.tensor([False, False])
  failed = update_failure(failed, torch.tensor([0.0, -1.0]), torch.zeros(2), 0.02)
  assert failed.tolist() == [True, False]
  failed = update_failure(failed, torch.tensor([-1.0, -1.0]), torch.zeros(2), 0.02)
  assert failed.tolist() == [True, False]


def test_sustained_trunk_contact_is_failure():
  failed = update_failure(
    torch.zeros(2, dtype=torch.bool),
    torch.tensor([-1.0, -1.0]),
    torch.tensor([24.0, 25.0]),
    0.02,
  )
  assert failed.tolist() == [False, True]


def test_command_schedules_include_stop_reverse_and_full_yaw_range():
  assert scheduled_command("maneuver", 10.0) == (0.0, 0.0, 0.0)
  assert scheduled_command("maneuver", 25.0) == (-0.25, 0.0, 0.0)
  assert scheduled_command("yaw", 8.0) == (0.0, 0.0, 1.0)
  assert scheduled_command("yaw", 24.0) == (0.0, 0.0, -1.0)
  assert scheduled_command("maneuver", 40.0) == scheduled_command("maneuver", 0.0)
  assert scheduled_command("yaw", 64.0) == scheduled_command("yaw", 24.0)


def test_isolated_terrain_padding_does_not_fill_inverted_pit():
  cfg = FixedDifficultyTerrainCfg(
    terrain=pyramid_stairs_inv(step_height_range=(0.18, 0.18), step_width=0.3),
    difficulty=1.0,
    size=(64.0, 64.0),
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(0.0, spec, np.random.default_rng(43))
  assert output.origin[0] == 32.0 and output.origin[1] == 32.0
  assert output.origin[2] < -1.0
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  def hit_z(x, y):
    distance = mujoco.mj_ray(
      model,
      data,
      np.array([x, y, 5.0]),
      np.array([0.0, 0.0, -1.0]),
      None,
      True,
      -1,
      np.zeros(1, dtype=np.int32),
    )
    return 5.0 - distance

  assert hit_z(32.0, 32.0) == pytest.approx(output.origin[2])
  assert hit_z(45.0, 32.0) == pytest.approx(0.0)
