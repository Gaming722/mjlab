"""Tests for the Go1 D435-like perceptive locomotion configuration."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationTermCfg
from mjlab.sensor import (
  CameraSensorCfg,
  GridPatternCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastData,
  RayCastSensorCfg,
)
from mjlab.tasks.velocity.config.go1.env_cfgs import (
  D435_FOVY_DEG,
  D435_HEIGHT,
  D435_WIDTH,
  unitree_go1_blind_env_cfg,
  unitree_go1_edge_perceptive_env_cfg,
  unitree_go1_guarded_smooth_env_cfg,
  unitree_go1_motor_rehearsal_env_cfg,
  unitree_go1_perceptive_env_cfg,
  unitree_go1_precise_edge_perceptive_env_cfg,
  unitree_go1_robust_perceptive_env_cfg,
  unitree_go1_smooth_perceptive_env_cfg,
  unitree_go1_stairs_low_speed_env_cfg,
  unitree_go1_stairs_perceptive_env_cfg,
)
from mjlab.tasks.velocity.config.go1.rl_cfg import (
  unitree_go1_edge_ppo_runner_cfg,
  unitree_go1_perceptive_ppo_runner_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.observations import depth_camera
from mjlab.terrains.config import GO1_STAIRS_TERRAINS_CFG
from mjlab.terrains.primitive_terrains import BoxPyramidStairsTerrainCfg


def test_stairs_task_preserves_checkpoint_contract_and_isolates_terrain():
  rough = unitree_go1_perceptive_env_cfg()
  stairs = unitree_go1_stairs_perceptive_env_cfg()
  play = unitree_go1_stairs_perceptive_env_cfg(play=True)
  assert "illegal_contact" in rough.terminations
  assert "thigh_collision" not in rough.rewards
  for cfg in (stairs, play):
    assert "illegal_contact" not in cfg.terminations
    penalty = cfg.rewards["thigh_collision"]
    assert penalty.weight == -0.1
    assert penalty.params["sensor_name"] == "thigh_ground_touch"
    assert any(s.name == "thigh_ground_touch" for s in cfg.scene.sensors or ())
    assert cfg.rewards["track_linear_velocity"].params["include_vertical"] is False
    assert cfg.rewards["track_angular_velocity"].params["include_roll_pitch"] is False
    assert cfg.rewards["track_angular_velocity"].params["yaw_in_heading_frame"] is True
    assert cfg.rewards["track_angular_velocity"].params["std"] == 0.25
    assert "command_vel" not in cfg.curriculum
    assert cfg.rewards["motion_deficit"].weight == -3.0
  assert "motion_deficit" not in rough.rewards
  assert "fell_over" in stairs.terminations
  assert "fell_over" not in rough.terminations
  assert stairs.scene.terrain is not None
  assert play.scene.terrain is not None
  assert rough.scene.terrain is not None
  terrain = stairs.scene.terrain.terrain_generator
  assert terrain is not None
  assert terrain.curriculum
  assert stairs.scene.terrain.max_init_terrain_level == 4
  assert set(stairs.observations) == set(rough.observations)
  for group in stairs.observations:
    assert set(stairs.observations[group].terms) == set(rough.observations[group].terms)
  assert play.scene.terrain.terrain_generator is not None
  play_terrain = play.scene.terrain.terrain_generator
  assert play_terrain.num_rows == 1
  assert play_terrain.difficulty_range == (1.0, 1.0)
  assert set(play_terrain.sub_terrains) == {"pyramid_stairs", "pyramid_stairs_inv"}
  for name, stair in play_terrain.sub_terrains.items():
    assert stair == terrain.sub_terrains[name]
  assert play.scene.terrain.max_init_terrain_level == 0
  assert not play.curriculum
  assert "randomize_terrain" in play.events
  assert play.scene.num_envs == 1
  assert play.sim.nconmax is None
  terrain.sub_terrains["flat"].proportion = 0.9
  assert GO1_STAIRS_TERRAINS_CFG.sub_terrains["flat"].proportion == 0.1
  assert rough.scene.terrain.terrain_generator is not None
  assert rough.scene.terrain.terrain_generator.sub_terrains["flat"].proportion == 0.2


def test_low_speed_stairs_focus_preserves_general_task():
  cfg = unitree_go1_stairs_low_speed_env_cfg()
  general = unitree_go1_stairs_perceptive_env_cfg()
  assert cfg.scene.terrain is not None
  assert general.scene.terrain is not None
  terrain = cfg.scene.terrain.terrain_generator
  assert terrain is not None
  assert terrain.num_rows == 7
  assert cfg.scene.terrain.max_init_terrain_level == 6
  assert terrain.sub_terrains["pyramid_stairs_inv"].proportion == 0.6
  assert "push_robot" not in cfg.events
  assert "push_robot" in general.events
  assert general.scene.terrain.max_init_terrain_level == 4
  assert cfg.scene.num_envs == 4096
  assert cfg.observations == general.observations


def _ray_sensor(distances: torch.Tensor):
  cfg = RayCastSensorCfg(
    name="depth",
    frame=ObjRef(type="site", name="camera", entity="robot"),
    pattern=PinholeCameraPatternCfg(width=3, height=1, fovy=60.0),
  )
  sensor = cfg.build()
  zeros3 = torch.zeros(distances.shape[0], distances.shape[1], 3)
  sensor._cached_data = RayCastData(  # noqa: SLF001
    distances=distances,
    normals_w=zeros3,
    hit_pos_w=zeros3,
    pos_w=torch.zeros(distances.shape[0], 3),
    quat_w=torch.zeros(distances.shape[0], 4),
    frame_pos_w=torch.zeros(distances.shape[0], 1, 3),
    frame_quat_w=torch.zeros(distances.shape[0], 1, 4),
  )
  sensor._cache_valid = True  # noqa: SLF001
  return sensor


def _term(env) -> depth_camera:
  cfg = ObservationTermCfg(
    func=depth_camera,
    params={
      "sensor_name": "depth",
      "width": 3,
      "height": 1,
      "fovy": 60.0,
      "near": 0.1,
      "far": 2.0,
      "policy_hz": 50,
      "sensor_hz": 30,
    },
  )
  return depth_camera(cfg, cast(ManagerBasedRlEnv, env))


def _call(term: depth_camera, env) -> torch.Tensor:
  return term(cast(ManagerBasedRlEnv, env), sensor_name="depth")


def test_raycast_depth_converts_range_to_optical_z_and_saturates_misses():
  distances = torch.tensor([[1.0, 1.0, -1.0]])
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    common_step_counter=0,
    scene={"depth": _ray_sensor(distances)},
  )
  result = _call(_term(env), env)

  assert result.shape == (1, 1, 1, 3)
  assert result[0, 0, 0, 1].item() == pytest.approx(1.0)
  assert result[0, 0, 0, 0].item() < 1.0
  assert result[0, 0, 0, 2].item() == pytest.approx(2.0)


def test_depth_capture_follows_30_hz_schedule_for_50_hz_policy():
  distances = torch.ones(1, 3)
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    common_step_counter=0,
    scene={"depth": _ray_sensor(distances)},
  )
  term = _term(env)
  center_values = []
  for step in range(6):
    env.common_step_counter = step
    env.scene["depth"]._cached_data.distances.fill_(0.2 * (step + 1))  # noqa: SLF001
    center_values.append(_call(term, env)[0, 0, 0, 1].item())

  assert center_values == pytest.approx([0.2, 0.2, 0.6, 0.6, 1.0, 1.2])


def test_depth_partial_reset_refreshes_only_reset_environment():
  distances = torch.tensor([[1.0, 1.0, 1.0], [1.5, 1.5, 1.5]])
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    common_step_counter=0,
    scene={"depth": _ray_sensor(distances)},
  )
  term = _term(env)
  _call(term, env)
  env.scene["depth"]._cached_data.distances[:] = torch.tensor(  # noqa: SLF001
    [[1.2, 1.2, 1.2], [1.8, 1.8, 1.8]]
  )
  term.reset(torch.tensor([1]))

  result = _call(term, env)
  assert result[:, 0, 0, 1].tolist() == pytest.approx([1.0, 1.8])


def test_perceptive_configs_keep_blind_actor_and_privileged_critic_contract():
  blind = unitree_go1_blind_env_cfg()
  cfg = unitree_go1_perceptive_env_cfg()

  assert set(cfg.observations["actor"].terms) == set(blind.observations["actor"].terms)
  assert "base_lin_vel" not in cfg.observations["actor"].terms
  assert "height_scan" not in cfg.observations["actor"].terms
  assert "base_lin_vel" in cfg.observations["critic"].terms
  assert "height_scan" in cfg.observations["critic"].terms
  assert set(cfg.observations["depth"].terms) == {"depth"}

  sensor = cfg.scene.sensors[-1]
  assert isinstance(sensor, RayCastSensorCfg)
  assert isinstance(sensor.pattern, PinholeCameraPatternCfg)
  assert (sensor.pattern.width, sensor.pattern.height) == (D435_WIDTH, D435_HEIGHT)
  assert sensor.pattern.fovy == pytest.approx(D435_FOVY_DEG)
  assert cfg.scene.num_envs == 4096


def test_camera_validation_config_and_asymmetric_runner():
  cfg = unitree_go1_perceptive_env_cfg(camera_sensor=True)
  sensor = cfg.scene.sensors[-1]
  assert isinstance(sensor, CameraSensorCfg)
  assert sensor.data_types == ("depth",)
  assert (sensor.width, sensor.height) == (D435_WIDTH, D435_HEIGHT)
  assert cfg.scene.num_envs == 16

  runner = unitree_go1_perceptive_ppo_runner_cfg()
  assert runner.actor.class_name.endswith(":SpatialSoftmaxCNNModel")
  assert runner.actor.cnn_cfg is not None
  assert runner.critic.class_name == "MLPModel"
  assert runner.obs_groups == {
    "actor": ("actor", "depth"),
    "critic": ("critic",),
  }


def test_robust_task_regularizes_base_and_covers_all_rough_types():
  cfg = unitree_go1_robust_perceptive_env_cfg()
  play = unitree_go1_robust_perceptive_env_cfg(play=True)
  for env in (cfg, play):
    assert env.rewards["lin_vel_z"].weight == -8.0
    assert env.rewards["ang_vel_xy"].weight == -0.2
    command = env.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    assert command.ranges.ang_vel_z == (-1.0, 1.0)
    assert command.rel_turning_envs == 0.3
    assert env.scene.terrain is not None
    terrain = env.scene.terrain.terrain_generator
    assert terrain is not None
    assert len(terrain.sub_terrains) == 7
    assert "fell_over" in env.terminations
  assert cfg.curriculum["terrain_levels"].params["traversal_term"] == "tile_traversed"
  assert cfg.terminations["tile_traversed"].time_out
  assert cfg.terminations["tile_traversed"].params["threshold_fraction"] == 1.8
  assert cfg.episode_length_s == 30.0
  assert "tile_traversed" not in play.terminations
  for env in (cfg, play, unitree_go1_stairs_perceptive_env_cfg(play=True)):
    assert env.rewards["feet_edge"].weight == -1.0
    scan = next(s for s in env.scene.sensors or () if s.name == "terrain_scan")
    assert isinstance(scan, RayCastSensorCfg)
    assert isinstance(scan.pattern, GridPatternCfg)
    assert scan.pattern.resolution == 0.025
    assert not scan.debug_vis_initial
    assert env.observations["critic"].terms["height_scan"].params["grid_stride"] == 4


def test_edge_finetune_covers_varied_treads_without_mutating_robust_task():
  cfg = unitree_go1_edge_perceptive_env_cfg()
  play = unitree_go1_edge_perceptive_env_cfg(play=True)
  for env in (cfg, play):
    assert env.scene.terrain is not None
    terrain = env.scene.terrain.terrain_generator
    assert terrain is not None
    stairs = [
      s
      for s in terrain.sub_terrains.values()
      if isinstance(s, BoxPyramidStairsTerrainCfg)
    ]
    assert len(stairs) == 6
    assert {s.step_width for s in stairs} == {0.27, 0.30, 0.33}
    assert sum(s.proportion for s in stairs) == pytest.approx(0.6)
    assert terrain.sub_terrains["flat"].proportion == pytest.approx(0.2)
    assert env.rewards["feet_edge"].params["landing_weight"] == 20.0
    assert env.rewards["feet_edge"].params["support_weight"] == 1.0
    assert len(terrain.sub_terrains) == 11
    assert sum(s.proportion for s in terrain.sub_terrains.values()) == pytest.approx(1)
    assert set(env.observations["actor"].terms) == set(cfg.observations["actor"].terms)
  original = unitree_go1_robust_perceptive_env_cfg()
  assert original.scene.terrain is not None
  assert original.scene.terrain.terrain_generator is not None
  assert len(original.scene.terrain.terrain_generator.sub_terrains) == 7
  rl = unitree_go1_edge_ppo_runner_cfg()
  assert rl.algorithm.schedule == "fixed"
  assert rl.algorithm.learning_rate == 5e-5


def test_precise_edge_profile_appends_critic_inputs_and_preserves_actor_contract():
  original = unitree_go1_edge_perceptive_env_cfg()
  critic_names = list(original.observations["critic"].terms)
  for play in (False, True):
    original = unitree_go1_edge_perceptive_env_cfg(play=play)
    cfg = unitree_go1_precise_edge_perceptive_env_cfg(play=play)
    assert list(cfg.observations["critic"].terms) == [
      *critic_names,
      "feet_edge_proximity",
    ]
    assert cfg.observations["actor"] == original.observations["actor"]
    assert cfg.observations["depth"] == original.observations["depth"]
    assert cfg.actions == original.actions
    proximity = cfg.observations["critic"].terms["feet_edge_proximity"]
    assert proximity.params["margin"] == cfg.rewards["feet_edge"].params["margin"]
    assert cfg.rewards["lin_vel_z"].weight == original.rewards["lin_vel_z"].weight
  assert "feet_edge_proximity" not in original.observations["critic"].terms


def test_smooth_profile_preserves_actor_and_enables_substep_motor_sampling():
  original = unitree_go1_precise_edge_perceptive_env_cfg()
  cfg = unitree_go1_smooth_perceptive_env_cfg()
  assert cfg.observations == original.observations
  assert cfg.commands == original.commands
  action = cfg.actions["joint_pos"]
  previous = original.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  assert isinstance(previous, JointPositionActionCfg)
  assert action.scale == previous.scale
  assert action.clip == previous.clip
  assert cfg.metrics["motor_motion"].per_substep
  assert cfg.rewards["torque_limit"].params["soft_ratio"] < 1.0
  assert "torque_limit" not in original.rewards


def test_guarded_smooth_profile_preserves_safety_costs_and_demands_progress():
  for play in (False, True):
    original = unitree_go1_smooth_perceptive_env_cfg(play=play)
    cfg = unitree_go1_guarded_smooth_env_cfg(play=play)
    assert cfg.observations == original.observations
    assert cfg.commands == original.commands
    assert cfg.actions == original.actions
    for name in (
      "torque_limit",
      "actual_torque_limit",
      "joint_acc_l2",
      "base_lin_acc_l2",
      "base_ang_acc_l2",
      "feet_edge",
    ):
      assert cfg.rewards[name] == original.rewards[name]
    assert not cfg.terminations["predicted_effort_guard"].time_out
    assert "predicted_effort_guard" not in original.terminations
    assert cfg.rewards["motion_deficit"].weight == -3.0
    assert cfg.rewards["motion_deficit"].params["minimum_speed_fraction"] == 0.5


def test_motor_rehearsal_preserves_control_and_revisits_all_difficulty_levels():
  for play in (False, True):
    original = unitree_go1_guarded_smooth_env_cfg(play=play)
    cfg = unitree_go1_motor_rehearsal_env_cfg(play=play)
    assert cfg.observations == original.observations
    assert cfg.actions == original.actions
    assert cfg.commands == original.commands
    assert cfg.sim == original.sim
    assert "predicted_effort_guard" not in cfg.terminations
    assert "predicted_effort_guard" in original.terminations
    assert "terrain_levels" not in cfg.curriculum
    assert cfg.scene.terrain is not None
    assert original.scene.terrain is not None
    terrain = cfg.scene.terrain.terrain_generator
    previous = original.scene.terrain.terrain_generator
    assert terrain is not None and previous is not None
    assert terrain.curriculum
    assert cfg.scene.terrain.max_init_terrain_level == terrain.num_rows - 1
    assert sum(s.proportion for s in terrain.sub_terrains.values()) == pytest.approx(1)
    for name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
      assert terrain.sub_terrains[name].proportion == 0.1
      assert previous.sub_terrains[name].proportion == 0.025
    for name, sub in terrain.sub_terrains.items():
      # Only the sampling weights change; difficulty and geometry stay intact.
      expected = dict(vars(previous.sub_terrains[name]))
      actual = dict(vars(sub))
      expected.pop("proportion")
      actual.pop("proportion")
      assert actual == expected
    for name in (
      "joint_acc_l2",
      "base_lin_acc_l2",
      "base_ang_acc_l2",
      "lin_vel_z",
      "ang_vel_xy",
      "actual_torque_limit",
      "feet_edge",
    ):
      assert cfg.rewards[name] == original.rewards[name]
    assert cfg.rewards["torque_limit"].params["hard_violation_cost"] == 10
