"""Requested motor effort must not be hidden by simulated force saturation."""

from types import SimpleNamespace
from typing import cast

import mujoco
import pytest
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.velocity.mdp.motor_safety import (
  MonitoredJointPositionActionCfg,
  effort_limit_cost,
  predicted_pd_effort,
)


@pytest.fixture
def motor_env():
  def spec_fn():
    return mujoco.MjSpec.from_string("""
      <mujoco><worldbody><body name="base"><freejoint/>
        <geom type="sphere" size="0.1" mass="1"/>
        <body name="link"><joint name="joint0" range="-4 4"/>
          <geom type="capsule" size="0.02 0.1" mass="0.1"/>
        </body>
      </body></worldbody></mujoco>
    """)

  entity = Entity(
    EntityCfg(
      spec_fn=spec_fn,
      articulation=EntityArticulationInfoCfg(
        actuators=(
          BuiltinPositionActuatorCfg(
            target_names_expr=("joint0",),
            stiffness=10.0,
            damping=1.0,
            effort_limit=2.0,
          ),
        )
      ),
    )
  )
  model = entity.compile()
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device="cpu")
  entity.initialize(model, sim.model, sim.data, "cpu")
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    physics_dt=0.01,
    scene={"robot": entity},
    sim=sim,
    extras={"log": {}},
  )
  return cast(ManagerBasedRlEnv, env), entity


def make_action(env, bounds=None):
  return MonitoredJointPositionActionCfg(
    entity_name="robot",
    actuator_names=("joint0",),
    use_default_offset=False,
    guard_position_limits=bounds,
  ).build(env)


def write_state(entity, positions, velocities):
  entity.data.write_joint_state(
    torch.tensor(positions).reshape(2, 1), torch.tensor(velocities).reshape(2, 1)
  )


def test_limit_cost_sees_unsaturated_request_and_per_joint_limits():
  torque = predicted_pd_effort(
    torch.tensor([[3.0, 10.0]]),
    torch.zeros(1, 2),
    torch.zeros(1, 2),
    torch.full((1, 2), 10.0),
    torch.ones(1, 2),
  )
  limit = torch.tensor([23.7, 35.55])
  assert (torque > limit).all()
  ratio = torque.abs() / limit
  assert (
    effort_limit_cost(ratio).item() > effort_limit_cost(torch.ones_like(ratio)).item()
  )
  assert effort_limit_cost(torch.full_like(ratio, 0.8)).item() == 0
  assert effort_limit_cost(torch.full_like(ratio, 0.9)).item() > 0
  excess = torch.tensor([[1.001, 0.2]])
  torch.testing.assert_close(
    effort_limit_cost(excess, hard_violation_cost=1.0), effort_limit_cost(excess) + 1.0
  )


def test_initial_peak_survives_relaxed_post_step_state_without_changing_target(
  motor_env,
):
  env, robot = motor_env
  action = make_action(env)
  write_state(robot, [0.0, 0.0], [0.0, 0.0])
  target = torch.tensor([[3.0], [0.1]])
  action.process_actions(target)
  torch.testing.assert_close(
    action.initial_guard_ratio[:, 0], torch.tensor([15.0, 0.5])
  )
  write_state(robot, [3.0, 0.1], [0.0, 0.0])
  action.apply_actions()
  torch.testing.assert_close(action.peak_effort_ratio[:, 0], torch.tensor([15.0, 0.5]))
  torch.testing.assert_close(robot.data.joint_pos_target, target)


def test_sdk_target_clip_and_damping_can_exceed_when_raw_request_cancels(motor_env):
  env, robot = motor_env
  action = make_action(env, {"joint0": (-0.5, 0.5)})
  write_state(robot, [0.0, 0.0], [30.0, -30.0])
  action.process_actions(torch.tensor([[3.0], [-3.0]]))
  torch.testing.assert_close(action.peak_request_ratio, torch.zeros(2, 1))
  torch.testing.assert_close(action.initial_guard_ratio, torch.full((2, 1), 12.5))
  action.apply_actions()
  torch.testing.assert_close(robot.data.joint_pos_target, torch.tensor([[3.0], [-3.0]]))


def test_held_command_velocity_peak_is_not_lost(motor_env):
  env, robot = motor_env
  action = make_action(env)
  write_state(robot, [0.0, 0.0], [0.0, 0.0])
  action.process_actions(torch.zeros(2, 1))
  action.apply_actions()
  write_state(robot, [0.0, 0.0], [4.0, 0.0])
  action.apply_actions()
  write_state(robot, [0.0, 0.0], [0.0, 0.0])
  action.apply_actions()
  torch.testing.assert_close(action.peak_guard_ratio[:, 0], torch.tensor([2.0, 0.0]))
  assert action.guard_exceeded.tolist() == [True, False]


def test_encoder_coordinates_and_randomized_runtime_gains(motor_env):
  env, robot = motor_env
  action = make_action(env)
  write_state(robot, [0.0, 0.0], [0.0, 0.0])
  robot.data.encoder_bias[:, 0] = 0.1
  action.process_actions(torch.full((2, 1), 0.2))
  action.apply_actions()
  torch.testing.assert_close(action.initial_guard_ratio, torch.full((2, 1), 0.5))
  torch.testing.assert_close(robot.data.joint_pos_target, torch.full((2, 1), 0.1))
  env.sim.expand_model_fields(["actuator_gainprm", "actuator_biasprm"])
  env.sim.model.actuator_gainprm[:, :, 0] = 20.0
  env.sim.model.actuator_biasprm[:, :, 1] = -20.0
  action.process_actions(torch.full((2, 1), 0.2))
  torch.testing.assert_close(action.initial_guard_ratio, torch.full((2, 1), 0.5))
  torch.testing.assert_close(action.peak_request_ratio, torch.ones(2, 1))


def test_actual_torque_peak_and_substep_average_are_both_retained(motor_env):
  env, _ = motor_env
  action = make_action(env)
  action.process_actions(torch.zeros(2, 1))
  action.apply_actions()
  env.sim.data.actuator_force[:] = torch.tensor([[2.0], [0.5]])
  action.record_substep_motion()
  env.sim.data.actuator_force[:] = torch.tensor([[0.0], [0.5]])
  action.record_substep_motion()
  torch.testing.assert_close(action.peak_actual_ratio[:, 0], torch.tensor([1.0, 0.25]))
  torch.testing.assert_close(action.peak_saturation_ratio, action.peak_actual_ratio)
  torch.testing.assert_close(action.near_limit_sample_count, torch.tensor([1.0, 0.0]))
  torch.testing.assert_close(action.saturation_sample_count, torch.tensor([1.0, 0.0]))
  torch.testing.assert_close(
    action.actual_effort_sq_sum[:, 0] / action.acc_sample_count,
    torch.tensor([2.0, 0.25]),
  )


def test_motion_intervals_exclude_reset_and_external_velocity_push(motor_env):
  env, robot = motor_env
  action = make_action(env)
  write_state(robot, [0.0, 0.0], [7.0, 7.0])
  action.process_actions(torch.zeros(2, 1))
  action.apply_actions()
  write_state(robot, [0.0, 0.0], [7.1, 7.1])
  action.record_substep_motion()
  torch.testing.assert_close(
    action.joint_acc_sq_sum, torch.full((2, 1), 100.0), atol=0.001, rtol=0.001
  )
  action.apply_actions()
  write_state(robot, [0.0, 0.0], [7.2, 7.2])
  action.record_substep_motion()
  torch.testing.assert_close(
    action.joint_acc_sq_sum, torch.full((2, 1), 200.0), atol=0.001, rtol=0.001
  )
  action.reset(torch.tensor([0]))
  assert action.acc_sample_count.tolist() == [0.0, 2.0]
  write_state(robot, [0.0, 0.0], [-50.0, 100.0])
  action.process_actions(torch.zeros(2, 1))
  action.apply_actions()
  action.record_substep_motion()
  torch.testing.assert_close(action.joint_acc_sq_sum, torch.zeros(2, 1))
  action.apply_actions()
  write_state(robot, [0.0, 0.0], [-49.9, 100.1])
  action.record_substep_motion()
  torch.testing.assert_close(
    action.joint_acc_sq_sum, torch.full((2, 1), 100.0), atol=0.001, rtol=0.001
  )
  torch.testing.assert_close(action.acc_sample_count, torch.tensor([2.0, 2.0]))


def test_world_acceleration_does_not_count_rotation_of_body_coordinates(motor_env):
  env, robot = motor_env
  action = make_action(env)
  data = robot.data
  data.write_root_pose(torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).repeat(2, 1))
  velocity = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0]]).repeat(2, 1)
  data.write_root_velocity(velocity)
  action.process_actions(torch.zeros(2, 1))
  action.apply_actions()
  s = 2**-0.5
  data.write_root_pose(torch.tensor([[0.0, 0.0, 0.0, s, 0.0, 0.0, s]]).repeat(2, 1))
  data.write_root_velocity(velocity)
  action.record_substep_motion()
  torch.testing.assert_close(
    action.base_lin_acc_sq_sum, torch.zeros(2, 3), atol=1e-8, rtol=0
  )
  torch.testing.assert_close(
    action.base_ang_acc_sq_sum, torch.zeros(2, 3), atol=1e-8, rtol=0
  )
