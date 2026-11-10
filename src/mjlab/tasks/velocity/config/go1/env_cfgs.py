"""Unitree Go1 velocity environment configurations."""

import math
from copy import deepcopy
from dataclasses import asdict
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
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  CameraSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import GO1_STAIRS_TERRAINS_CFG, ROUGH_TERRAINS_CFG
from mjlab.terrains.primitive_terrains import BoxPyramidStairsTerrainCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg

TerrainType = Literal["rough", "obstacles"]

D435_DEPTH_POSITION = (0.27542987, 0.0175, 0.12552532)
D435_MUJOCO_QUATERNION = (0.612372, 0.353553, -0.353553, -0.612372)
D435_WIDTH = 80
D435_HEIGHT = 45
D435_HFOV = math.radians(87.0)
D435_FOVY_DEG = math.degrees(
  2.0 * math.atan(math.tan(D435_HFOV / 2.0) * D435_HEIGHT / D435_WIDTH)
)


def _configure_feet_edge(cfg: ManagerBasedRlEnvCfg) -> None:
  terrain_scan = next(s for s in cfg.scene.sensors or () if s.name == "terrain_scan")
  assert isinstance(terrain_scan, RayCastSensorCfg)
  assert isinstance(terrain_scan.pattern, GridPatternCfg)
  terrain_scan.pattern.resolution = 0.025
  # Keep the critic's original 17x11 scan and checkpoint input dimensions.
  cfg.observations["critic"].terms["height_scan"].params["grid_stride"] = 4
  # The edge term already draws the scan. Keep other ray clouds available in
  # the viewer without obscuring the colored edge overlay by default.
  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, RayCastSensorCfg) and sensor.name in (
      "terrain_scan",
      "foot_height_scan",
      "d435_depth",
    ):
      sensor.debug_vis_initial = False
  cfg.rewards["feet_edge"] = RewardTermCfg(
    func=mdp.feet_edge,
    weight=-1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "terrain_scan",
      "asset_cfg": SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL")),
      "margin": 0.05,
      "height_threshold": 0.03,
      "landing_weight": 10.0,
      "support_weight": 0.25,
    },
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
    cfg.scene.num_envs = 4096
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
  # Allow brief thigh contact during stair exploration without ending the episode.
  cfg.terminations.pop("illegal_contact", None)
  cfg.rewards["thigh_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": "thigh_ground_touch"},
  )
  # Prioritize commanded planar motion over holding a quiet nominal posture.
  # Vertical and roll/pitch motion are necessary when negotiating stair edges.
  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_linear_velocity"].params.update(std=0.25, include_vertical=False)
  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.rewards["track_angular_velocity"].params["include_roll_pitch"] = False
  cfg.rewards["track_angular_velocity"].params.update(
    std=0.25, yaw_in_heading_frame=True
  )
  cfg.rewards["pose"].weight = 0.3
  cfg.rewards["action_rate_l2"].weight = -0.03
  cfg.rewards["foot_clearance"].weight = -0.5
  cfg.rewards["foot_swing_height"].weight = -0.1
  cfg.rewards["motion_deficit"] = RewardTermCfg(
    func=mdp.commanded_motion_deficit,
    weight=-3.0,
    params={"command_name": "twist", "minimum_speed_fraction": 0.5},
  )
  # Keep a consistent speed distribution as terrain difficulty increases.
  cfg.curriculum.pop("command_vel", None)
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.ranges.lin_vel_x = (-0.5, 1.0)
  command.ranges.lin_vel_y = (-0.3, 0.3)
  command.rel_forward_envs = 0.5
  command.rel_standing_envs = 0.05
  if not play:
    cfg.terminations["fell_over"] = TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    )
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
  if play:
    _configure_feet_edge(cfg)
  return cfg


def unitree_go1_stairs_low_speed_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Practice frontal low-speed ascent after learning general stair locomotion."""
  cfg = unitree_go1_stairs_perceptive_env_cfg(play=play)
  if play:
    return cfg
  assert cfg.scene.terrain is not None
  terrain = cfg.scene.terrain.terrain_generator
  assert terrain is not None
  terrain.num_rows = 7
  cfg.scene.terrain.max_init_terrain_level = 6
  for name in ("pyramid_stairs", "pyramid_stairs_inv"):
    stair = terrain.sub_terrains[name]
    assert isinstance(stair, BoxPyramidStairsTerrainCfg)
    stair.step_height_range = (0.12, 0.18)
  proportions = {
    "pyramid_stairs": 0.25,
    "pyramid_stairs_inv": 0.6,
    "flat": 0.1,
    "random_rough": 0.025,
    "hf_pyramid_slope": 0.025,
  }
  for name, proportion in proportions.items():
    terrain.sub_terrains[name].proportion = proportion
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.ranges.lin_vel_x = (-0.3, 0.6)
  command.ranges.lin_vel_y = (-0.2, 0.2)
  command.ranges.ang_vel_z = (-0.3, 0.3)
  command.rel_forward_envs = 0.8
  cfg.events["reset_base"].params["pose_range"].update(
    x=(-0.2, 0.2), y=(-0.2, 0.2), yaw=(-0.15, 0.15)
  )
  cfg.events.pop("push_robot", None)
  return cfg


def unitree_go1_robust_perceptive_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Train balanced terrain locomotion with explicit base-motion regularization."""
  cfg = unitree_go1_stairs_perceptive_env_cfg(play=play)
  _configure_feet_edge(cfg)
  cfg.rewards["lin_vel_z"] = RewardTermCfg(
    func=mdp.base_vertical_velocity_l2, weight=-8.0
  )
  cfg.rewards["ang_vel_xy"] = RewardTermCfg(
    func=mdp.base_roll_pitch_velocity_l2, weight=-0.2
  )
  cfg.rewards["forward_tipping"] = RewardTermCfg(
    func=mdp.forward_tipping_cost,
    weight=-2.0,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["rough_turn_support"] = RewardTermCfg(
    func=mdp.rough_turn_support_cost,
    weight=-1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "command_name": "twist",
    },
  )
  for name in ("foot_clearance", "foot_swing_height"):
    cfg.rewards[name].params["rough_turn_extra_height"] = 0.05
  cfg.rewards["pose"].weight = 0.5
  cfg.rewards["action_rate_l2"].weight = -0.08
  cfg.rewards["action_acc_l2"] = RewardTermCfg(func=mdp.action_acc_l2, weight=-0.01)
  cfg.rewards["joint_acc_l2"] = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=mdp.joint_torques_l2, weight=-2e-4
  )
  cfg.rewards["soft_landing"].weight = -5e-5
  cfg.rewards["motion_deficit"].weight = -1.0
  cfg.rewards["motion_deficit"].params["minimum_speed_fraction"] = 0.35
  cfg.rewards["termination"] = RewardTermCfg(func=mdp.is_terminated, weight=-50.0)
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.ranges.lin_vel_x = (-0.5, 0.8)
  command.ranges.lin_vel_y = (-0.3, 0.3)
  command.ranges.ang_vel_z = (-1.0, 1.0)
  command.rel_forward_envs = 0.35
  command.rel_turning_envs = 0.3
  command.rel_standing_envs = 0.1
  command.resampling_time_range = (4.0, 8.0)
  assert cfg.scene.terrain is not None
  terrain = deepcopy(ROUGH_TERRAINS_CFG)
  terrain.curriculum = True
  terrain.num_rows = 16
  terrain.num_cols = len(terrain.sub_terrains)
  proportions = {
    "flat": 0.15,
    "pyramid_stairs": 0.2,
    "pyramid_stairs_inv": 0.2,
    "hf_pyramid_slope": 0.1,
    "hf_pyramid_slope_inv": 0.1,
    "random_rough": 0.15,
    "wave_terrain": 0.1,
  }
  for name, proportion in proportions.items():
    terrain.sub_terrains[name].proportion = proportion
  for name in ("pyramid_stairs", "pyramid_stairs_inv"):
    stair = terrain.sub_terrains[name]
    assert isinstance(stair, BoxPyramidStairsTerrainCfg)
    stair.step_height_range = (0.02, 0.18)
    stair.step_width = 0.3
  cfg.scene.terrain.terrain_generator = terrain
  cfg.scene.terrain.max_init_terrain_level = 8
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
  )
  cfg.terminations["trunk_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "trunk_ground_touch", "force_threshold": 100.0},
  )
  if not play:
    # Keep collecting post-descent transitions instead of ending at the first
    # patch edge. At moderate speeds this covers an adjacent terrain as well.
    cfg.episode_length_s = 30.0
    cfg.events["reset_base"].params["pose_range"].update(x=(-0.2, 0.2), y=(-0.2, 0.2))
    cfg.terminations["tile_traversed"] = TerminationTermCfg(
      func=mdp.terrain_edge_reached,
      params={"threshold_fraction": 1.8},
      time_out=True,
    )
    cfg.curriculum["terrain_levels"].params["traversal_term"] = "tile_traversed"
    cfg.events["push_robot"].interval_range_s = (5.0, 10.0)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.2, 0.2),
      "y": (-0.2, 0.2),
      "z": (-0.1, 0.1),
      "roll": (-0.2, 0.2),
      "pitch": (-0.2, 0.2),
      "yaw": (-0.3, 0.3),
    }
  return cfg


def unitree_go1_edge_perceptive_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Practice edge-aware placement with varied treads and frequent stair exposure."""
  cfg = unitree_go1_robust_perceptive_env_cfg(play=play)
  cfg.rewards["feet_edge"].params.update(landing_weight=20.0, support_weight=1.0)
  assert cfg.scene.terrain is not None
  terrain = cfg.scene.terrain.terrain_generator
  assert terrain is not None
  variants: dict[str, SubTerrainCfg] = {}
  other_weights = {
    "flat": 0.2,
    "random_rough": 0.1,
    "wave_terrain": 0.05,
    "hf_pyramid_slope": 0.025,
    "hf_pyramid_slope_inv": 0.025,
  }
  for name, sub in terrain.sub_terrains.items():
    if isinstance(sub, BoxPyramidStairsTerrainCfg):
      for width in (0.27, 0.30, 0.33):
        variant = deepcopy(sub)
        variant.step_width = width
        variant.platform_width = 1.0
        variant.proportion = 0.30 / 3
        variants[f"{name}_w{round(width * 100):02d}"] = variant
    else:
      variant = deepcopy(sub)
      variant.proportion = other_weights[name]
      variants[name] = variant
  terrain.sub_terrains = variants
  terrain.num_cols = len(variants)
  cfg.scene.terrain.max_init_terrain_level = terrain.num_rows - 1
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.rel_forward_envs = 0.50
  command.rel_turning_envs = 0.25
  command.rel_standing_envs = 0.10
  command.resampling_time_range = (6.0, 10.0)
  return cfg


def unitree_go1_precise_edge_perceptive_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Resume edge-aware critics with a wider safety band and calmer base motion."""
  cfg = unitree_go1_edge_perceptive_env_cfg(play=play)
  edge = cfg.rewards["feet_edge"]
  edge.params["margin"] = 0.07
  cfg.rewards["ang_vel_xy"].weight = -0.5
  # Append only: converted 259-input critics preserve their existing function.
  cfg.observations["critic"].terms["feet_edge_proximity"] = ObservationTermCfg(
    func=mdp.feet_edge_proximity,
    params={
      "height_sensor_name": "terrain_scan",
      "asset_cfg": SceneEntityCfg("robot", site_names=("FR", "FL", "RR", "RL")),
      "margin": edge.params["margin"],
      "height_threshold": edge.params["height_threshold"],
    },
  )
  return cfg


def configure_go1_motor_monitor(cfg: ManagerBasedRlEnvCfg) -> None:
  """Observe the Go1 SDK's pre-saturation guard without modifying actions."""
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  if not isinstance(action, mdp.MonitoredJointPositionActionCfg):
    cfg.actions["joint_pos"] = mdp.MonitoredJointPositionActionCfg(
      **asdict(action),
      guard_position_limits={
        ".*_hip_joint": (-1.047, 1.047),
        ".*_thigh_joint": (-0.663, 2.966),
        ".*_calf_joint": (-2.721, -0.837),
      },
    )
  cfg.metrics["motor_motion"] = MetricsTermCfg(
    func=mdp.sample_motor_motion,
    params={"action_name": "joint_pos"},
    per_substep=True,
  )


def unitree_go1_smooth_perceptive_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Refine a capable policy for PD effort margins and smooth physical motion."""
  cfg = unitree_go1_precise_edge_perceptive_env_cfg(play=play)
  configure_go1_motor_monitor(cfg)
  cfg.rewards["torque_limit"] = RewardTermCfg(
    func=mdp.torque_limit,
    weight=-20.0,
    params={
      "action_name": "joint_pos",
      "soft_ratio": 0.85,
      "hard_excess_scale": 5.0,
      "hard_violation_cost": 1.0,
    },
  )
  cfg.rewards["actual_torque_limit"] = RewardTermCfg(
    func=mdp.actual_torque_limit,
    weight=-10.0,
    params={"action_name": "joint_pos", "soft_ratio": 0.8},
  )
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=mdp.monitored_joint_torques_l2,
    weight=-4e-4,
    params={"action_name": "joint_pos"},
  )
  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=mdp.monitored_joint_acceleration_l2,
    weight=-2e-6,
    params={"action_name": "joint_pos"},
  )
  cfg.rewards["base_lin_acc_l2"] = RewardTermCfg(
    func=mdp.base_linear_acceleration_l2,
    weight=-0.005,
    params={"action_name": "joint_pos"},
  )
  cfg.rewards["base_ang_acc_l2"] = RewardTermCfg(
    func=mdp.base_angular_acceleration_l2,
    weight=-0.0002,
    params={"action_name": "joint_pos"},
  )
  cfg.rewards["action_rate_l2"].weight = -0.12
  cfg.rewards["action_acc_l2"].weight = -0.03
  return cfg


def unitree_go1_guarded_smooth_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Refine smooth locomotion without continuing past a deployment guard trip."""
  cfg = unitree_go1_smooth_perceptive_env_cfg(play=play)
  cfg.terminations["predicted_effort_guard"] = TerminationTermCfg(
    func=mdp.predicted_effort_guard, params={"action_name": "joint_pos"}
  )
  # A guard trip is terminal on hardware. Counter the alternative of parking
  # before difficult terrain by retaining a strong directed-motion objective.
  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["motion_deficit"].weight = -3.0
  cfg.rewards["motion_deficit"].params["minimum_speed_fraction"] = 0.5
  return cfg


def unitree_go1_motor_rehearsal_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rehearse the full terrain range with torque margins and recovery practice."""
  cfg = unitree_go1_guarded_smooth_env_cfg(play=play)
  # Preserve recovery examples while making even short effort violations costly.
  # The continuous evaluator still counts every deployment guard trip as failure.
  cfg.terminations.pop("predicted_effort_guard")
  cfg.rewards["torque_limit"].params["hard_violation_cost"] = 10.0
  cfg.rewards["termination"].weight = -400.0
  cfg.rewards["track_angular_velocity"].weight = 4.0
  cfg.rewards["track_angular_velocity"].params["turning_std"] = 0.225
  # Retain leg smoothness as the policy restores full-range yaw tracking.
  cfg.rewards["joint_acc_l2"].weight = -2.75e-6
  # Practice slow approaches and stop/resume transitions without rewarding
  # stationary stair poses as a close approximation to a small forward command.
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.forward_min_velocity = 0.2
  command.rel_standing_envs = 0.15
  command.resampling_time_range = (4.0, 8.0)
  cfg.rewards["track_linear_velocity"].params["std"] = 0.15
  cfg.rewards["motion_deficit"].params["minimum_speed_fraction"] = 0.8
  cfg.rewards["motion_deficit"].weight = -4.0
  # Retain edge placement while adapting to torque margins. Increase touchdown
  # cost only; do not make ongoing support or slow forward progress costlier.
  cfg.rewards["feet_edge"].params["landing_weight"] = 30.0
  # Do not let an already capable policy forget hard cases as the new costs
  # demote their curriculum levels. Keep the initial full-range level samples.
  cfg.curriculum.pop("terrain_levels", None)
  assert cfg.scene.terrain is not None
  terrain = cfg.scene.terrain.terrain_generator
  assert terrain is not None
  cfg.scene.terrain.max_init_terrain_level = terrain.num_rows - 1
  for name, sub in terrain.sub_terrains.items():
    if name.startswith("pyramid_stairs_inv_w"):
      sub.proportion = 0.30 / 3
    elif name.startswith("pyramid_stairs_w"):
      sub.proportion = 0.20 / 3
    else:
      sub.proportion = {
        "flat": 0.15,
        "random_rough": 0.10,
        "wave_terrain": 0.05,
        "hf_pyramid_slope": 0.10,
        "hf_pyramid_slope_inv": 0.10,
      }[name]
  return cfg


def unitree_go1_stair_depth_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rehearse narrow treads independently of height, plus coupled difficulty."""
  cfg = unitree_go1_motor_rehearsal_env_cfg(play=play)
  assert cfg.scene.terrain is not None
  terrain = cfg.scene.terrain.terrain_generator
  assert terrain is not None
  variants: dict[str, SubTerrainCfg] = {
    name: deepcopy(sub)
    for name, sub in terrain.sub_terrains.items()
    if not isinstance(sub, BoxPyramidStairsTerrainCfg)
  }
  # Keep the original terrain-family exposure and all existing reward weights.
  # Fixed narrow treads cover low/narrow cases that a coupled curve would miss.
  for family, total in (("pyramid_stairs", 0.20), ("pyramid_stairs_inv", 0.30)):
    template = terrain.sub_terrains[f"{family}_w30"]
    assert isinstance(template, BoxPyramidStairsTerrainCfg)
    for width, fraction in (
      (0.22, 0.25),
      (0.225, 0.15),
      (0.24, 0.15),
      (0.27, 0.15),
      (0.30, 0.10),
      (0.33, 0.10),
    ):
      variant = deepcopy(template)
      variant.step_width = width
      variant.proportion = total * fraction
      variants[f"{family}_d{round(width * 1000)}"] = variant
    coupled = deepcopy(template)
    coupled.step_width_range = (0.30, 0.22)
    coupled.proportion = total * 0.10
    variants[f"{family}_adaptive"] = coupled
  terrain.sub_terrains = variants
  terrain.num_cols = len(variants)
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
