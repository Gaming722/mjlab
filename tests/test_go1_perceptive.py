"""Tests for the Go1 D435-like perceptive locomotion configuration."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import ObservationTermCfg
from mjlab.sensor import (
  CameraSensorCfg,
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
  unitree_go1_perceptive_env_cfg,
  unitree_go1_stairs_perceptive_env_cfg,
)
from mjlab.tasks.velocity.config.go1.rl_cfg import (
  unitree_go1_perceptive_ppo_runner_cfg,
)
from mjlab.tasks.velocity.mdp.observations import depth_camera
from mjlab.terrains.config import GO1_STAIRS_TERRAINS_CFG


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
