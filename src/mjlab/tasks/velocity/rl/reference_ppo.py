# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause
# See reference_ppo.LICENSE for the retained upstream license.

"""Fixed-rate feedforward PPO with an optional frozen-reference action loss.

The PPO update follows rsl-rl's BSD-3-Clause implementation (ETH Zurich and
NVIDIA, 2021-2026). This opt-in variant deliberately rejects extensions whose
update semantics it does not implement; ordinary tasks continue to use RSL PPO.
"""

import hashlib
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


def masked_reference_loss(
  mean: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
  """Mean squared action error over replay samples, with zero empty-mask loss."""
  if mean.ndim != 2 or mean.shape != target.shape or mask.shape != (mean.shape[0], 1):
    raise ValueError("Reference targets/mask must match the action batch.")
  if not torch.all((mask == 0) | (mask == 1)):
    raise ValueError("Reference mask must be binary and finite.")
  error = (mean - target.detach()).square().mean(dim=-1, keepdim=True)
  return (error * mask).sum() / mask.sum().clamp_min(1)


class ReferencePolicyPPO(PPO):
  """Constrain action means on replay terrain without changing the rewards."""

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: RolloutStorage,
    reference_coefficient: float = 0.0,
    reference_checkpoint: str | None = None,
    reference_sha256: str | None = None,
    reference_mask_key: str = "reference_mask",
    reference_resume_coefficient: float | None = None,
    **kwargs: Any,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    if not math.isfinite(reference_coefficient) or reference_coefficient < 0:
      raise ValueError("reference_coefficient must be finite and nonnegative.")
    self.reference_coefficient = reference_coefficient
    if reference_resume_coefficient is not None and (
      not math.isfinite(reference_resume_coefficient)
      or reference_resume_coefficient < 0
    ):
      raise ValueError("reference_resume_coefficient must be finite and nonnegative.")
    self.reference_resume_coefficient = reference_resume_coefficient
    self.reference_mask_key = reference_mask_key
    self.reference_sha256 = reference_sha256
    self.reference_actor: MLPModel | None = None
    if reference_coefficient == 0:
      return
    if (
      self.schedule != "fixed"
      or actor.is_recurrent
      or critic.is_recurrent
      or self.rnd is not None
      or self.symmetry is not None
      or self.is_multi_gpu
      or self.use_mixed_precision
      or not isinstance(actor.distribution, GaussianDistribution)
    ):
      raise ValueError("Reference PPO requires fixed-rate, single-GPU Gaussian PPO.")
    if reference_checkpoint is None or reference_sha256 is None:
      raise ValueError("Reference checkpoint and SHA256 are required.")
    path = Path(reference_checkpoint)
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference_sha256:
      raise ValueError("Reference checkpoint SHA256 mismatch.")
    self.reference_actor = deepcopy(actor).eval()
    state = torch.load(path, map_location=self.device, weights_only=False)
    self.reference_actor.load_state_dict(state["actor_state_dict"], strict=True)
    self.reference_actor.requires_grad_(False)

  def update(self) -> dict[str, float]:
    if self.reference_coefficient == 0:
      return super().update()
    assert self.reference_actor is not None
    totals = dict(value=0.0, surrogate=0.0, entropy=0.0, reference=0.0)
    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for batch in generator:
      assert batch.observations is not None and batch.actions is not None
      assert batch.advantages is not None and batch.old_actions_log_prob is not None
      assert batch.values is not None and batch.returns is not None
      advantages = batch.advantages
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
      self.actor(batch.observations, stochastic_output=True)
      log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations)
      entropy = self.actor.output_entropy
      mean = self.actor.output_distribution_params[0]
      with torch.no_grad():
        target = self.reference_actor(batch.observations)
      mask = batch.observations[self.reference_mask_key]
      assert isinstance(mask, torch.Tensor)
      reference = masked_reference_loss(mean, target, mask)
      ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze())
      advantage = advantages.squeeze()
      surrogate = torch.maximum(
        -advantage * ratio,
        -advantage * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
      ).mean()
      if self.use_clipped_value_loss:
        clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value = torch.maximum(
          (values - batch.returns).square(), (clipped - batch.returns).square()
        ).mean()
      else:
        value = (values - batch.returns).square().mean()
      loss = (
        surrogate
        + self.value_loss_coef * value
        - self.entropy_coef * entropy.mean()
        + self.reference_coefficient * reference
      )
      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      for key, term in (
        ("value", value),
        ("surrogate", surrogate),
        ("entropy", entropy.mean()),
        ("reference", reference),
      ):
        totals[key] += term.item()
    observations = self.storage.observations.flatten(0, 1)
    assert isinstance(observations, TensorDict)
    self.actor.update_normalization(observations)
    self.critic.update_normalization(observations)
    self.storage.clear()
    updates = self.num_learning_epochs * self.num_mini_batches
    return {key: value / updates for key, value in totals.items()}

  def save(self) -> dict:
    state = super().save()
    state["reference_policy"] = self._reference_contract()
    return state

  def _reference_contract(self) -> dict:
    return dict(
      sha256=self.reference_sha256,
      coefficient=self.reference_coefficient,
      mask_key=self.reference_mask_key,
    )

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    if (load_cfg is None or load_cfg.get("optimizer", False)) and (
      "reference_policy" in loaded_dict
    ):
      expected = self._reference_contract()
      previous = dict(expected)
      if self.reference_resume_coefficient is not None:
        previous["coefficient"] = self.reference_resume_coefficient
      if loaded_dict["reference_policy"] not in (expected, previous):
        raise ValueError("Resume reference policy contract differs from checkpoint.")
    return super().load(loaded_dict, load_cfg, strict)
