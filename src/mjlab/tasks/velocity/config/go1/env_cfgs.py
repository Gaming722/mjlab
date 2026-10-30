"""Unitree Go1 velocity environment configurations."""

import math
from copy import deepcopy
from typing import Literal

from mjlab.asset_zoo.robots import (
  GO1_ACTION_SCALE,
  get_go1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  CameraSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import GO1_STAIRS_TERRAINS_CFG

TerrainType = Literal["rough", "obstacles"]

D435_DEPTH_POSITION = (0.27542987, 0.0175, 0.12552532)
D435_MUJOCO_QUATERNION = (0.612372, 0.353553, -0.353553, -0.612372)
D435_WIDTH = 80
D435_HEIGHT = 45
D435_HFOV = math.radians(87.0)
D435_FOVY_DEG = math.degrees(
  2.0 * math.atan(math.tan(D435_HFOV / 2.0) * D435_HEIGHT / D435_WIDTH)
)


def unitree_go1_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go1_robot_cfg()}

  # Set raycast sensor frame to Go1 trunk.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "trunk"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  thigh_geom_names = tuple(
    f"{leg}_thigh_collision{i}" for leg in foot_names for i in (1, 2, 3)
  )
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=thigh_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  calf_geom_names = tuple(
    f"{leg}_calf_collision{i}" for leg in foot_names for i in (1, 2)
  )
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=calf_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  trunk_head_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=("trunk_collision", "head_collision"),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    thigh_ground_cfg,
    shank_ground_cfg,
    trunk_head_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO1_ACTION_SCALE

  cfg.viewer.body_name = "trunk"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Replace the base foot_friction with per-axis friction events for condim 6.
  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  # Per-body-group collision penalties.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": shank_ground_cfg.name},
  )
  cfg.rewards["trunk_head_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": trunk_head_ground_cfg.name},
  )

  # On rough terrain the quadruped tilts significantly; don't terminate on
  # orientation alone. Let out_of_terrain_bounds handle resets.
  cfg.terminations.pop("fell_over", None)

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # A single preview world cannot share contact capacity with other worlds.
    # Include contacts at the pre-reset pose in Warp's automatic allocation.
    cfg.sim.nconmax = None
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go1_blind_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Go1 rough terrain config with proprioceptive actor observations."""
  cfg = unitree_go1_rough_env_cfg(play=play)
  # Allow more solver iterations for rough terrain contacts.
  cfg.sim.mujoco.iterations = 50
  cfg.sim.mujoco.ls_iterations = 50
  del cfg.observations["actor"].terms["base_lin_vel"]
  del cfg.observations["actor"].terms["height_scan"]
  return cfg


def _d435_observation_cfg(play: bool) -> ObservationGroupCfg:
  return ObservationGroupCfg(
    terms={
      "depth": ObservationTermCfg(
        func=mdp.depth_camera,
        params={
          "sensor_name": "d435_depth",
          "width": D435_WIDTH,
          "height": D435_HEIGHT,
          "fovy": D435_FOVY_DEG,
          "near": 0.1,
          "far": 2.0,
          "policy_hz": 50,
          "sensor_hz": 30,
          "noise_std": 0.0 if play else 0.01,
          "dropout_prob": 0.0 if play else 0.005,
        },
        delay_min_lag=0,
        delay_max_lag=0 if play else 1,
        delay_hold_prob=0.8,
      )
    },
    concatenate_terms=True,
    enable_corruption=not play,
    nan_policy="error",
  )


def unitree_go1_perceptive_env_cfg(
  play: bool = False,
  camera_sensor: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a Go1 perceptive locomotion task with D435-like depth."""
  cfg = unitree_go1_blind_env_cfg(play=play)
  if camera_sensor:
    depth_sensor = CameraSensorCfg(
      name="d435_depth",
      parent_body="robot/trunk",
      pos=D435_DEPTH_POSITION,
      quat=D435_MUJOCO_QUATERNION,
      fovy=D435_FOVY_DEG,
      width=D435_WIDTH,
      height=D435_HEIGHT,
      data_types=("depth",),
      use_textures=False,
      use_shadows=False,
      enabled_geom_groups=(0, 2),
    )
    # The rendered camera is intended for low-count validation, not throughput runs.
    cfg.scene.num_envs = 16
  else:
    depth_sensor = RayCastSensorCfg(
      name="d435_depth",
      frame=ObjRef(type="site", name="d435_depth", entity="robot"),
      pattern=PinholeCameraPatternCfg(
        width=D435_WIDTH,
        height=D435_HEIGHT,
        fovy=D435_FOVY_DEG,
      ),
      ray_alignment="base",
      max_distance=3.0,
      exclude_parent_body=True,
      include_geom_groups=(0, 2),
      debug_vis=play,
    )
    # Dense pinhole scans allocate several per-ray GPU buffers. Tune this count
    # to the target GPU's memory and measured training throughput.
    cfg.scene.num_envs = 2048
    if not play:
      camera_site = SceneEntityCfg("robot", site_names=("d435_depth",))
      cfg.events["d435_mount_position"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.site_pos,
        params={
          "asset_cfg": camera_site,
          "operation": "add",
          "ranges": (-0.005, 0.005),
        },
      )
      cfg.events["d435_mount_orientation"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.site_quat,
        params={
          "asset_cfg": camera_site,
          "roll_range": (-math.radians(2.0), math.radians(2.0)),
          "pitch_range": (-math.radians(2.0), math.radians(2.0)),
          "yaw_range": (-math.radians(2.0), math.radians(2.0)),
        },
      )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_sensor,)
  cfg.observations["depth"] = _d435_observation_cfg(play)
  if play:
    cfg.scene.num_envs = 1
  return cfg


def unitree_go1_stairs_perceptive_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Fine-tune perceptive Go1 on a stair-focused 2--18 cm curriculum."""
  cfg = unitree_go1_perceptive_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  terrain = deepcopy(GO1_STAIRS_TERRAINS_CFG)
  # Begin at 2--6.3 cm; higher steps are reached through the curriculum.
  cfg.scene.terrain.max_init_terrain_level = 4
  if play:
    # Preview the hardest training stairs deterministically. Random resets may
    # switch stair direction, but cannot select easy rows or non-stair terrain.
    terrain.sub_terrains = {
      name: terrain.sub_terrains[name]
      for name in ("pyramid_stairs", "pyramid_stairs_inv")
    }
    terrain.num_rows = 1
    terrain.num_cols = 2
    terrain.difficulty_range = (1.0, 1.0)
    cfg.scene.terrain.max_init_terrain_level = 0
  cfg.scene.terrain.terrain_generator = terrain
  return cfg


def unitree_go1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 flat terrain velocity configuration."""
  cfg = unitree_go1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensors and collision sensors not needed on flat.
  remove_sensors = {
    "terrain_scan",
    "self_collision",
    "thigh_ground_touch",
    "shank_ground_touch",
    "trunk_ground_touch",
  }
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  # Remove granular collision rewards (not useful on flat ground).
  for key in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards.pop(key, None)

  # On flat terrain fell_over is sufficient; thigh contact implies fallen.
  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
