from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go1_blind_env_cfg,
  unitree_go1_flat_env_cfg,
  unitree_go1_perceptive_env_cfg,
  unitree_go1_rough_env_cfg,
  unitree_go1_stairs_low_speed_env_cfg,
  unitree_go1_stairs_perceptive_env_cfg,
)
from .rl_cfg import unitree_go1_perceptive_ppo_runner_cfg, unitree_go1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Stairs-LowSpeed-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_stairs_low_speed_env_cfg(),
  play_env_cfg=unitree_go1_stairs_low_speed_env_cfg(play=True),
  rl_cfg=unitree_go1_perceptive_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Stairs-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_stairs_perceptive_env_cfg(),
  play_env_cfg=unitree_go1_stairs_perceptive_env_cfg(play=True),
  rl_cfg=unitree_go1_perceptive_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-Go1",
  env_cfg=unitree_go1_rough_env_cfg(),
  play_env_cfg=unitree_go1_rough_env_cfg(play=True),
  rl_cfg=unitree_go1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Blind-Unitree-Go1",
  env_cfg=unitree_go1_blind_env_cfg(),
  play_env_cfg=unitree_go1_blind_env_cfg(play=True),
  rl_cfg=unitree_go1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go1",
  env_cfg=unitree_go1_flat_env_cfg(),
  play_env_cfg=unitree_go1_flat_env_cfg(play=True),
  rl_cfg=unitree_go1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_perceptive_env_cfg(),
  play_env_cfg=unitree_go1_perceptive_env_cfg(play=True),
  rl_cfg=unitree_go1_perceptive_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-Go1-Perceptive-Camera-Play",
  env_cfg=unitree_go1_perceptive_env_cfg(camera_sensor=True),
  play_env_cfg=unitree_go1_perceptive_env_cfg(play=True, camera_sensor=True),
  rl_cfg=unitree_go1_perceptive_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
