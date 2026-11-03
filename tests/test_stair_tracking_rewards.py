"""Stair tracking can permit vertical and roll/pitch transients."""

import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.rewards import (
  commanded_motion_deficit,
  track_angular_velocity,
  track_linear_velocity,
)
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz


@pytest.mark.parametrize("include_transients", [True, False])
def test_tracking_optional_stair_transients(include_transients):
  data = SimpleNamespace(
    root_link_lin_vel_b=torch.tensor([[0.5, 0.0, 0.5]]),
    root_link_ang_vel_b=torch.tensor([[0.5, 0.5, 0.0]]),
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={"robot": SimpleNamespace(data=data)},
      command_manager=SimpleNamespace(
        get_command=lambda _: torch.tensor([[0.5, 0.0, 0.0]])
      ),
    ),
  )
  linear = track_linear_velocity(
    env, std=0.5, command_name="twist", include_vertical=include_transients
  )
  angular = track_angular_velocity(
    env, std=0.5, command_name="twist", include_roll_pitch=include_transients
  )
  assert linear.item() == pytest.approx(
    torch.exp(torch.tensor(-1.0)).item() if include_transients else 1.0
  )
  assert angular.item() == pytest.approx(
    torch.exp(torch.tensor(-2.0)).item() if include_transients else 1.0
  )


def test_motion_deficit_requires_progress_along_command():
  command = torch.tensor(
    [
      [0.0, 0.0, 1.0],
      [0.3, 0.0, 0.0],
      [0.3, 0.0, 0.0],
      [0.3, 0.0, 0.0],
      [0.3, 0.0, 0.0],
      [-0.3, 0.0, 0.0],
    ]
  )
  velocity = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.075, 0.0, 0.0],
      [0.15, 0.0, 0.0],
      [0.0, 0.3, 0.0],
      [-0.15, 0.0, 0.0],
    ]
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={
        "robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=velocity))
      },
      command_manager=SimpleNamespace(get_command=lambda _: command),
    ),
  )
  cost = commanded_motion_deficit(env, command_name="twist")
  torch.testing.assert_close(cost, torch.tensor([0.0, 1.0, 0.5, 0.0, 1.0, 0.0]))


def test_heading_rate_tracking_accounts_for_roll_pitch_coupling():
  roll, pitch = math.pi / 6, math.pi / 9
  quat = quat_from_euler_xyz(
    torch.tensor([roll]), torch.tensor([pitch]), torch.tensor([0.0])
  )
  omega_b = torch.tensor([[0.0, 1.0, 0.0]])
  expected_rate = math.sin(roll) / math.cos(pitch)
  data = SimpleNamespace(
    root_link_ang_vel_b=omega_b,
    root_link_ang_vel_w=quat_apply(quat, omega_b),
    root_link_quat_w=quat,
    forward_vec_b=torch.tensor([[1.0, 0.0, 0.0]]),
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={"robot": SimpleNamespace(data=data)},
      command_manager=SimpleNamespace(
        get_command=lambda _: torch.tensor([[0.0, 0.0, expected_rate]])
      ),
    ),
  )
  actual_heading = track_angular_velocity(
    env,
    std=0.5,
    command_name="twist",
    include_roll_pitch=False,
    yaw_in_heading_frame=True,
  )
  body_axis = track_angular_velocity(
    env,
    std=0.5,
    command_name="twist",
    include_roll_pitch=False,
  )
  assert actual_heading.item() == pytest.approx(1.0)
  assert body_axis.item() == pytest.approx(math.exp(-((expected_rate / 0.5) ** 2)))
