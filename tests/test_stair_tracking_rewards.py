"""Stair tracking can permit vertical and roll/pitch transients."""

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
