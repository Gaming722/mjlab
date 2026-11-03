import wandb
from rsl_rl.algorithms import PPO
from rsl_rl.utils import WandbLogWriter

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      is_wandb = isinstance(self.logger.writer, WandbLogWriter)
      run_name: str = wandb.run.name if is_wandb and wandb.run else "local"  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if is_wandb and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")


class VelocityFineTuneRunner(VelocityOnPolicyRunner):
  """Restore training state but apply the configured fixed fine-tuning rate."""

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    restore_optimizer = load_cfg is None or load_cfg.get("optimizer", False)
    if restore_optimizer and self.cfg["algorithm"]["schedule"] == "fixed":
      assert isinstance(self.alg, PPO)
      rate = self.cfg["algorithm"]["learning_rate"]
      self.alg.learning_rate = rate
      for group in self.alg.optimizer.param_groups:
        group["lr"] = rate
    return infos
