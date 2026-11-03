"""Continuous locomotion scoring must not stop checking after a passage."""

import torch

from mjlab.scripts.evaluate_go1_locomotion import scheduled_command, update_failure


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
