"""RL configuration for Unitree Go1 velocity task."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )


def unitree_go1_perceptive_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create asymmetric PPO config for D435 depth plus proprioception."""
  cfg = unitree_go1_ppo_runner_cfg()
  cfg.actor.class_name = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
  cfg.actor.hidden_dims = (256, 256, 128)
  cfg.actor.cnn_cfg = {
    "output_channels": [16, 32],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
  }
  cfg.algorithm.share_cnn_encoders = False
  cfg.obs_groups = {
    "actor": ("actor", "depth"),
    "critic": ("critic",),
  }
  cfg.experiment_name = "go1_perceptive_velocity"
  return cfg


def unitree_go1_edge_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Conservative updates for edge-placement fine-tuning of an existing policy."""
  cfg = unitree_go1_perceptive_ppo_runner_cfg()
  cfg.algorithm.learning_rate = 5e-5
  cfg.algorithm.schedule = "fixed"
  cfg.max_iterations = 600
  cfg.save_interval = 100
  return cfg


def unitree_go1_smooth_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Use small fixed updates when refining an already capable edge policy."""
  cfg = unitree_go1_edge_ppo_runner_cfg()
  cfg.algorithm.learning_rate = 2e-5
  cfg.max_iterations = 600
  return cfg


def unitree_go1_motor_rehearsal_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Use prolonged, small fixed updates for motor-margin refinement."""
  cfg = unitree_go1_smooth_ppo_runner_cfg()
  cfg.algorithm.learning_rate = 1e-5
  cfg.max_iterations = 2000
  cfg.save_interval = 250
  return cfg


def unitree_go1_stair_depth_rehearsal_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Make short, conservative refinements after depth adaptation."""
  cfg = unitree_go1_motor_rehearsal_ppo_runner_cfg()
  cfg.algorithm.learning_rate = 5e-6
  cfg.max_iterations = 400
  cfg.save_interval = 100
  return cfg


def unitree_go1_stair_depth_long_stop_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Refine coverage conservatively while retaining the existing policy contract."""
  cfg = unitree_go1_stair_depth_rehearsal_ppo_runner_cfg()
  cfg.algorithm.learning_rate = 2.5e-6
  return cfg


def unitree_go1_stair_depth_balanced_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Allow longer consolidation with small fixed policy updates."""
  cfg = unitree_go1_stair_depth_long_stop_ppo_runner_cfg()
  cfg.max_iterations = 1200
  cfg.save_interval = 200
  return cfg


@dataclass
class Go1ReferencePpoCfg(RslRlPpoAlgorithmCfg):
  reference_coefficient: float = 5.0
  reference_resume_coefficient: float | None = None
  reference_checkpoint: str = "checkpoints/go1_stair_depth_reference/model_14500.pt"
  reference_sha256: str = (
    "82696d39a39bfaa033983f1e953c0c95c584d1b8c904b3c6ee097b884cb86ded"
  )
  reference_mask_key: str = "reference_mask"


def unitree_go1_stair_depth_reference_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Retain verified actions on old terrain while refining narrow treads."""
  cfg = unitree_go1_stair_depth_balanced_ppo_runner_cfg()
  cfg.algorithm = Go1ReferencePpoCfg(**vars(cfg.algorithm))
  cfg.algorithm.class_name = "mjlab.tasks.velocity.rl.reference_ppo:ReferencePolicyPPO"
  cfg.save_interval = 100
  return cfg


def unitree_go1_stair_depth_restart_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Keep the verified replay anchor while concentrating restart practice."""
  cfg = unitree_go1_stair_depth_reference_ppo_runner_cfg()
  assert isinstance(cfg.algorithm, Go1ReferencePpoCfg)
  cfg.algorithm.reference_coefficient = 25.0
  cfg.max_iterations = 2000
  cfg.save_interval = 250
  return cfg


def unitree_go1_stair_depth_exploration_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Increase stochastic exploration without changing deterministic actions."""
  cfg = unitree_go1_stair_depth_restart_ppo_runner_cfg()
  assert cfg.actor.distribution_cfg is not None
  # A resumed checkpoint restores std parameters, so init_std alone has no effect.
  # The distribution clamps sampling std; deterministic inference stays the mean.
  cfg.actor.distribution_cfg["std_range"] = (0.55, 1e6)
  return cfg
