"""RL configuration for Unitree Go1 velocity task."""

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
  cfg.max_iterations = 400
  cfg.save_interval = 100
  return cfg
