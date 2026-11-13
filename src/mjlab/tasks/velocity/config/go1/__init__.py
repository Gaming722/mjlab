from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityFineTuneRunner, VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go1_blind_env_cfg,
  unitree_go1_edge_perceptive_env_cfg,
  unitree_go1_flat_env_cfg,
  unitree_go1_guarded_smooth_env_cfg,
  unitree_go1_motor_rehearsal_env_cfg,
  unitree_go1_perceptive_env_cfg,
  unitree_go1_precise_edge_perceptive_env_cfg,
  unitree_go1_robust_perceptive_env_cfg,
  unitree_go1_rough_env_cfg,
  unitree_go1_smooth_perceptive_env_cfg,
  unitree_go1_stair_depth_balanced_env_cfg,
  unitree_go1_stair_depth_env_cfg,
  unitree_go1_stair_depth_long_stop_env_cfg,
  unitree_go1_stair_depth_progress_env_cfg,
  unitree_go1_stair_depth_reference_env_cfg,
  unitree_go1_stair_depth_rehearsal_env_cfg,
  unitree_go1_stair_depth_restart_env_cfg,
  unitree_go1_stairs_low_speed_env_cfg,
  unitree_go1_stairs_perceptive_env_cfg,
)
from .rl_cfg import (
  unitree_go1_edge_ppo_runner_cfg,
  unitree_go1_motor_rehearsal_ppo_runner_cfg,
  unitree_go1_perceptive_ppo_runner_cfg,
  unitree_go1_ppo_runner_cfg,
  unitree_go1_smooth_ppo_runner_cfg,
  unitree_go1_stair_depth_balanced_ppo_runner_cfg,
  unitree_go1_stair_depth_exploration_ppo_runner_cfg,
  unitree_go1_stair_depth_long_stop_ppo_runner_cfg,
  unitree_go1_stair_depth_reference_ppo_runner_cfg,
  unitree_go1_stair_depth_rehearsal_ppo_runner_cfg,
  unitree_go1_stair_depth_restart_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Stair-Depth-Explore-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_stair_depth_progress_env_cfg(),
  play_env_cfg=unitree_go1_stair_depth_progress_env_cfg(play=True),
  rl_cfg=unitree_go1_stair_depth_exploration_ppo_runner_cfg(),
  runner_cls=VelocityFineTuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Stair-Depth-Progress-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_stair_depth_progress_env_cfg(),
  play_env_cfg=unitree_go1_stair_depth_progress_env_cfg(play=True),
  rl_cfg=unitree_go1_stair_depth_restart_ppo_runner_cfg(),
  runner_cls=VelocityFineTuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Stair-Depth-Restart-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_stair_depth_restart_env_cfg(),
  play_env_cfg=unitree_go1_stair_depth_restart_env_cfg(play=True),
  rl_cfg=unitree_go1_stair_depth_restart_ppo_runner_cfg(),
  runner_cls=VelocityFineTuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Stair-Depth-Reference-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_stair_depth_reference_env_cfg(),
  play_env_cfg=unitree_go1_stair_depth_reference_env_cfg(play=True),
  rl_cfg=unitree_go1_stair_depth_reference_ppo_runner_cfg(),
  runner_cls=VelocityFineTuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Stair-Depth-Balanced-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_stair_depth_balanced_env_cfg(),
  play_env_cfg=unitree_go1_stair_depth_balanced_env_cfg(play=True),
  rl_cfg=unitree_go1_stair_depth_balanced_ppo_runner_cfg(),
  runner_cls=VelocityFineTuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Stair-Depth-LongStop-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_stair_depth_long_stop_env_cfg(),
  play_env_cfg=unitree_go1_stair_depth_long_stop_env_cfg(play=True),
  rl_cfg=unitree_go1_stair_depth_long_stop_ppo_runner_cfg(),
  runner_cls=VelocityFineTuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Stair-Depth-Rehearsal-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_stair_depth_rehearsal_env_cfg(),
  play_env_cfg=unitree_go1_stair_depth_rehearsal_env_cfg(play=True),
  rl_cfg=unitree_go1_stair_depth_rehearsal_ppo_runner_cfg(),
  runner_cls=VelocityFineTuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Stair-Depth-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_stair_depth_env_cfg(),
  play_env_cfg=unitree_go1_stair_depth_env_cfg(play=True),
  rl_cfg=unitree_go1_motor_rehearsal_ppo_runner_cfg(),
  runner_cls=VelocityFineTuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Edges-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_edge_perceptive_env_cfg(),
  play_env_cfg=unitree_go1_edge_perceptive_env_cfg(play=True),
  rl_cfg=unitree_go1_edge_ppo_runner_cfg(),
  runner_cls=VelocityFineTuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Edges-Precise-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_precise_edge_perceptive_env_cfg(),
  play_env_cfg=unitree_go1_precise_edge_perceptive_env_cfg(play=True),
  rl_cfg=unitree_go1_edge_ppo_runner_cfg(),
  runner_cls=VelocityFineTuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Smooth-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_smooth_perceptive_env_cfg(),
  play_env_cfg=unitree_go1_smooth_perceptive_env_cfg(play=True),
  rl_cfg=unitree_go1_smooth_ppo_runner_cfg(),
  runner_cls=VelocityFineTuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Smooth-Guarded-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_guarded_smooth_env_cfg(),
  play_env_cfg=unitree_go1_guarded_smooth_env_cfg(play=True),
  rl_cfg=unitree_go1_smooth_ppo_runner_cfg(),
  runner_cls=VelocityFineTuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Smooth-Rehearsal-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_motor_rehearsal_env_cfg(),
  play_env_cfg=unitree_go1_motor_rehearsal_env_cfg(play=True),
  rl_cfg=unitree_go1_motor_rehearsal_ppo_runner_cfg(),
  runner_cls=VelocityFineTuneRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Robust-Unitree-Go1-Perceptive",
  env_cfg=unitree_go1_robust_perceptive_env_cfg(),
  play_env_cfg=unitree_go1_robust_perceptive_env_cfg(play=True),
  rl_cfg=unitree_go1_perceptive_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

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
