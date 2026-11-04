"""Monitor requested PD efforts and physical acceleration before saturation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def predicted_pd_effort(
  target: torch.Tensor,
  position: torch.Tensor,
  velocity: torch.Tensor,
  kp: torch.Tensor,
  kd: torch.Tensor,
) -> torch.Tensor:
  """Requested torque, before actuator force clipping."""
  return kp * (target - position) - kd * velocity


def effort_limit_cost(
  ratio: torch.Tensor, soft_ratio: float = 0.85, hard_excess_scale: float = 5.0
) -> torch.Tensor:
  """Build a safety margin and penalize each joint's actual limit exceedance."""
  if not 0 < soft_ratio <= 1 or hard_excess_scale < 0:
    raise ValueError("Require 0 < soft_ratio <= 1 and nonnegative excess scale")
  soft = (ratio - soft_ratio).clamp_min(0).square()
  hard = (ratio - 1.0).clamp_min(0)
  return (soft + hard_excess_scale * hard).sum(-1)


@dataclass(kw_only=True)
class MonitoredJointPositionActionCfg(JointPositionActionCfg):
  """Preserve position actions while recording PD requests and substep motion.

  Guard bounds describe the deployment controller's target-position clipping.
  They affect monitoring only; simulated actions are never clipped or filtered.
  """

  guard_position_limits: dict[str, tuple[float, float]] | None = None

  def build(self, env: ManagerBasedRlEnv) -> MonitoredJointPositionAction:
    return MonitoredJointPositionAction(self, env)


class MonitoredJointPositionAction(JointPositionAction):
  """Sample the new command and every held-command physics substep.

  Nominal gains reproduce the deployment guard, including encoder coordinates
  and optional SDK target clipping. Runtime gains additionally constrain the
  request seen by the simulated actuator. A per-substep metrics callback samples
  integrated qvel after physics. Each command starts a new motion interval, so
  resets and externally imposed velocity pushes do not become fake acceleration.
  """

  def __init__(self, cfg: MonitoredJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    model = env.sim.mj_model
    global_joints = self._entity.indexing.joint_ids[self.target_ids].cpu().numpy()
    ctrl_ids = []
    for joint in global_joints:
      matches = np.flatnonzero(
        (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        & (model.actuator_trnid[:, 0] == joint)
      )
      if len(matches) != 1:
        raise ValueError("PD monitoring requires one position actuator per joint")
      ctrl = int(matches[0])
      if (
        int(model.jnt_type[joint])
        not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        or model.actuator_dyntype[ctrl] != mujoco.mjtDyn.mjDYN_NONE
        or model.actuator_gaintype[ctrl] != mujoco.mjtGain.mjGAIN_FIXED
        or model.actuator_biastype[ctrl] != mujoco.mjtBias.mjBIAS_AFFINE
        or not np.allclose(model.actuator_gear[ctrl], [1, 0, 0, 0, 0, 0])
        or not model.actuator_forcelimited[ctrl]
        or not np.isclose(
          model.actuator_biasprm[ctrl, 1], -model.actuator_gainprm[ctrl, 0]
        )
        or model.actuator_gainprm[ctrl, 0] <= 0
      ):
        raise ValueError(
          "PD monitoring requires force-limited unit-gear position actuators"
        )
      ctrl_ids.append(ctrl)
    self.ctrl_ids = torch.tensor(ctrl_ids, device=self.device, dtype=torch.long)
    self.nominal_kp = torch.tensor(
      model.actuator_gainprm[ctrl_ids, 0].copy(),
      device=self.device,
      dtype=torch.float32,
    )
    self.nominal_kd = torch.tensor(
      -model.actuator_biasprm[ctrl_ids, 2].copy(),
      device=self.device,
      dtype=torch.float32,
    )
    limits = model.actuator_forcerange[ctrl_ids]
    self.effort_limits = torch.tensor(
      np.minimum(-limits[:, 0], limits[:, 1]), device=self.device, dtype=torch.float32
    )
    if torch.any(~torch.isfinite(self.effort_limits) | (self.effort_limits <= 0)):
      raise ValueError("PD monitoring requires positive bilateral effort limits")
    self.guard_lower = torch.full((self.action_dim,), -torch.inf, device=self.device)
    self.guard_upper = torch.full((self.action_dim,), torch.inf, device=self.device)
    if cfg.guard_position_limits is not None:
      ids, _, bounds = resolve_matching_names_values(
        cfg.guard_position_limits, self.target_names
      )
      if len(ids) != self.action_dim:
        raise ValueError("Guard position limits must cover every action target")
      self.guard_lower[ids] = torch.tensor([v[0] for v in bounds], device=self.device)
      self.guard_upper[ids] = torch.tensor([v[1] for v in bounds], device=self.device)
      if torch.any(self.guard_lower >= self.guard_upper):
        raise ValueError("Guard lower bounds must be below upper bounds")
    shape = (self.num_envs, self.action_dim)
    self.initial_guard_ratio = torch.zeros(shape, device=self.device)
    self.peak_guard_ratio = torch.zeros_like(self.initial_guard_ratio)
    self.peak_request_ratio = torch.zeros_like(self.initial_guard_ratio)
    self.joint_acc_sq_sum = torch.zeros_like(self.initial_guard_ratio)
    self.base_lin_acc_sq_sum = torch.zeros((self.num_envs, 3), device=self.device)
    self.base_ang_acc_sq_sum = torch.zeros_like(self.base_lin_acc_sq_sum)
    self.acc_sample_count = torch.zeros(self.num_envs, device=self.device)
    self.peak_joint_acc = torch.zeros_like(self.acc_sample_count)
    self.peak_base_lin_acc = torch.zeros_like(self.acc_sample_count)
    self.peak_base_ang_acc = torch.zeros_like(self.acc_sample_count)
    self._previous_joint_velocity = torch.zeros_like(self.initial_guard_ratio)
    self._previous_base_velocity = torch.zeros((self.num_envs, 6), device=self.device)
    self._history_valid = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._physics_dt = env.physics_dt

  @property
  def peak_effort_ratio(self) -> torch.Tensor:
    return torch.maximum(self.peak_guard_ratio, self.peak_request_ratio)

  @property
  def guard_exceeded(self) -> torch.Tensor:
    """Match the deployment guard's 1e-5 Nm comparison tolerance."""
    invalid = ~torch.isfinite(self.peak_guard_ratio)
    return (invalid | (self.peak_guard_ratio > 1.0 + 1e-5 / self.effort_limits)).any(-1)

  def _effort_ratios(self) -> tuple[torch.Tensor, torch.Tensor]:
    data = self._entity.data
    position = data.joint_pos[:, self.target_ids]
    velocity = data.joint_vel[:, self.target_ids]
    bias = data.encoder_bias[:, self.target_ids]
    guard_target = self._processed_actions.clamp(self.guard_lower, self.guard_upper)
    guard = predicted_pd_effort(
      guard_target, position + bias, velocity, self.nominal_kp, self.nominal_kd
    )
    model = self._env.sim.model
    kp = model.actuator_gainprm[:, self.ctrl_ids, 0]
    kd = -model.actuator_biasprm[:, self.ctrl_ids, 2]
    request = predicted_pd_effort(
      self._processed_actions - bias, position, velocity, kp, kd
    )
    bounds = model.actuator_forcerange[:, self.ctrl_ids]
    runtime_limits = torch.minimum(-bounds[..., 0], bounds[..., 1])
    return guard.abs() / self.effort_limits, request.abs() / runtime_limits

  def _base_velocity_world(self) -> torch.Tensor:
    data = self._entity.data
    indices = self._entity.indexing
    if len(indices.free_joint_v_adr) != 6:
      return torch.zeros_like(self._previous_base_velocity)
    velocity = data.data.qvel[:, indices.free_joint_v_adr]
    orientation = data.data.qpos[:, indices.free_joint_q_adr[3:7]]
    return torch.cat((velocity[:, :3], quat_apply(orientation, velocity[:, 3:])), -1)

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)
    guard, request = self._effort_ratios()
    self.initial_guard_ratio.copy_(guard)
    self.peak_guard_ratio.copy_(guard)
    self.peak_request_ratio.copy_(request)
    self.joint_acc_sq_sum.zero_()
    self.base_lin_acc_sq_sum.zero_()
    self.base_ang_acc_sq_sum.zero_()
    self.acc_sample_count.zero_()
    self.peak_joint_acc.zero_()
    self.peak_base_lin_acc.zero_()
    self.peak_base_ang_acc.zero_()
    self._previous_joint_velocity.copy_(self._entity.data.joint_vel[:, self.target_ids])
    self._previous_base_velocity.copy_(self._base_velocity_world())
    self._history_valid.fill_(True)

  def apply_actions(self) -> None:
    guard, request = self._effort_ratios()
    self.peak_guard_ratio.copy_(torch.maximum(self.peak_guard_ratio, guard))
    self.peak_request_ratio.copy_(torch.maximum(self.peak_request_ratio, request))
    super().apply_actions()

  def record_substep_motion(self) -> torch.Tensor:
    """Called after integration, before events/reset; return mean joint accel²."""
    guard, request = self._effort_ratios()
    self.peak_guard_ratio.copy_(torch.maximum(self.peak_guard_ratio, guard))
    self.peak_request_ratio.copy_(torch.maximum(self.peak_request_ratio, request))
    joint_velocity = self._entity.data.joint_vel[:, self.target_ids]
    base_velocity = self._base_velocity_world()
    valid = self._history_valid.float()
    joint_acc = (joint_velocity - self._previous_joint_velocity) / self._physics_dt
    base_acc = (base_velocity - self._previous_base_velocity) / self._physics_dt
    self.joint_acc_sq_sum.add_(joint_acc.square() * valid[:, None])
    self.base_lin_acc_sq_sum.add_(base_acc[:, :3].square() * valid[:, None])
    self.base_ang_acc_sq_sum.add_(base_acc[:, 3:].square() * valid[:, None])
    self.peak_joint_acc.copy_(
      torch.maximum(self.peak_joint_acc, joint_acc.abs().amax(-1) * valid)
    )
    self.peak_base_lin_acc.copy_(
      torch.maximum(self.peak_base_lin_acc, base_acc[:, :3].norm(dim=-1) * valid)
    )
    self.peak_base_ang_acc.copy_(
      torch.maximum(self.peak_base_ang_acc, base_acc[:, 3:].norm(dim=-1) * valid)
    )
    self.acc_sample_count.add_(valid)
    self._previous_joint_velocity.copy_(joint_velocity)
    self._previous_base_velocity.copy_(base_velocity)
    self._history_valid.fill_(True)
    return joint_acc.square().mean(-1) * valid

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._history_valid[ids] = False
    for buffer in (
      self.initial_guard_ratio,
      self.peak_guard_ratio,
      self.peak_request_ratio,
      self.joint_acc_sq_sum,
      self.base_lin_acc_sq_sum,
      self.base_ang_acc_sq_sum,
      self.acc_sample_count,
      self.peak_joint_acc,
      self.peak_base_lin_acc,
      self.peak_base_ang_acc,
    ):
      buffer[ids] = 0


def motor_monitor(
  env: ManagerBasedRlEnv, action_name: str
) -> MonitoredJointPositionAction:
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, MonitoredJointPositionAction):
    raise TypeError("Motor safety rewards require MonitoredJointPositionAction")
  return term


def torque_limit(
  env: ManagerBasedRlEnv,
  action_name: str,
  soft_ratio: float = 0.85,
  hard_excess_scale: float = 5.0,
) -> torch.Tensor:
  monitor = motor_monitor(env, action_name)
  ratio = torch.nan_to_num(
    monitor.peak_effort_ratio, nan=10.0, posinf=10.0, neginf=10.0
  )
  env.extras["log"]["Metrics/motor/max_effort_ratio"] = ratio.amax()
  env.extras["log"]["Metrics/motor/guard_exceeded_fraction"] = (
    monitor.guard_exceeded.float().mean()
  )
  return effort_limit_cost(ratio, soft_ratio, hard_excess_scale)


def sample_motor_motion(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """Configure as a per-substep metric to collect complete physical intervals."""
  return motor_monitor(env, action_name).record_substep_motion()


def monitored_joint_acceleration_l2(
  env: ManagerBasedRlEnv, action_name: str
) -> torch.Tensor:
  monitor = motor_monitor(env, action_name)
  return monitor.joint_acc_sq_sum.sum(-1) / monitor.acc_sample_count.clamp_min(1)


def base_linear_acceleration_l2(
  env: ManagerBasedRlEnv, action_name: str
) -> torch.Tensor:
  monitor = motor_monitor(env, action_name)
  return monitor.base_lin_acc_sq_sum.sum(-1) / monitor.acc_sample_count.clamp_min(1)


def base_angular_acceleration_l2(
  env: ManagerBasedRlEnv, action_name: str
) -> torch.Tensor:
  monitor = motor_monitor(env, action_name)
  return monitor.base_ang_acc_sq_sum.sum(-1) / monitor.acc_sample_count.clamp_min(1)
