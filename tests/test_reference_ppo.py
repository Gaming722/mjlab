"""Reference regularization preserves PPO and isolates replay sample gradients."""

import hashlib
from copy import deepcopy
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import ObservationTermCfg
from mjlab.tasks.velocity.mdp.reference_policy import reference_terrain_mask
from mjlab.tasks.velocity.rl.reference_ppo import (
  ReferencePolicyPPO,
  masked_reference_loss,
)


def _algorithm(tmp_path, coefficient=3.0, mask=1.0, resume_coefficient=None):
  torch.manual_seed(41)
  obs = TensorDict(
    {"actor": torch.randn(8, 3), "reference_mask": torch.full((8, 1), mask)},
    batch_size=[8],
  )
  groups = {"actor": ["actor"], "critic": ["actor"]}
  actor = MLPModel(
    obs,
    groups,
    "actor",
    2,
    hidden_dims=(8,),
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
  )
  critic = MLPModel(obs, groups, "critic", 1, hidden_dims=(8,), obs_normalization=True)
  reference = deepcopy(actor)
  with torch.no_grad():
    for param in reference.mlp.parameters():
      param.add_(0.1)
  path = tmp_path / "reference.pt"
  torch.save({"actor_state_dict": reference.state_dict()}, path)
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  storage = RolloutStorage("rl", 8, 3, obs, [2], "cpu")
  algorithm = ReferencePolicyPPO(
    actor,
    critic,
    storage,
    reference_coefficient=coefficient,
    reference_resume_coefficient=resume_coefficient,
    reference_checkpoint=str(path),
    reference_sha256=digest,
    schedule="fixed",
    num_learning_epochs=2,
    num_mini_batches=2,
    learning_rate=3e-4,
  )
  with torch.no_grad():
    for _ in range(3):
      algorithm.act(obs)
      algorithm.process_env_step(
        obs, torch.randn(8), torch.zeros(8, dtype=torch.bool), {}
      )
    algorithm.compute_returns(obs)
  base = PPO(
    deepcopy(actor),
    deepcopy(critic),
    deepcopy(storage),
    schedule="fixed",
    num_learning_epochs=2,
    num_mini_batches=2,
    learning_rate=3e-4,
  )
  return algorithm, base


@pytest.mark.parametrize("coefficient,mask", [(0.0, 1.0), (3.0, 0.0)])
def test_reference_disabled_or_empty_replay_matches_original_ppo(
  tmp_path, coefficient, mask
):
  algorithm, base = _algorithm(tmp_path, coefficient, mask)
  torch.manual_seed(71)
  expected = base.update()
  torch.manual_seed(71)
  actual = algorithm.update()
  for name, value in expected.items():
    assert actual[name] == pytest.approx(value, abs=1e-6)
  for model, reference in (
    (algorithm.actor, base.actor),
    (algorithm.critic, base.critic),
  ):
    for name, value in model.state_dict().items():
      torch.testing.assert_close(
        value, reference.state_dict()[name], atol=1e-7, rtol=1e-6
      )
  assert algorithm.storage.step == base.storage.step == 0


def test_reference_update_changes_student_but_never_teacher(tmp_path):
  algorithm, base = _algorithm(tmp_path)
  assert algorithm.reference_actor is not None
  before = {k: v.clone() for k, v in algorithm.reference_actor.state_dict().items()}
  algorithm.train_mode()
  assert not algorithm.reference_actor.training
  torch.manual_seed(71)
  base.update()
  torch.manual_seed(71)
  losses = algorithm.update()
  assert losses["reference"] > 0
  assert any(
    not torch.equal(p, q)
    for p, q in zip(algorithm.actor.parameters(), base.actor.parameters(), strict=True)
  )
  for name, value in algorithm.reference_actor.state_dict().items():
    torch.testing.assert_close(value, before[name], atol=0, rtol=0)
  assert all(
    p.grad is None and not p.requires_grad
    for p in algorithm.reference_actor.parameters()
  )


def test_reference_loss_gradients_only_selected_samples():
  mean = torch.ones(3, 2, requires_grad=True)
  target = torch.zeros_like(mean, requires_grad=True)
  loss = masked_reference_loss(mean, target, torch.tensor([[1.0], [0.0], [1.0]]))
  loss.backward()
  assert mean.grad is not None
  torch.testing.assert_close(mean.grad[1], torch.zeros(2))
  assert (mean.grad[[0, 2]] > 0).all() and target.grad is None
  with pytest.raises(ValueError, match="binary"):
    masked_reference_loss(mean, target, torch.full((3, 1), float("nan")))


def test_reference_checkpoint_and_resume_contract_are_verified(tmp_path):
  algorithm, _ = _algorithm(tmp_path)
  state = algorithm.save()
  state["reference_policy"]["sha256"] = "wrong"
  with pytest.raises(ValueError, match="contract"):
    algorithm.load(state, None, True)
  algorithm.load(state, {"actor": True}, True)
  with pytest.raises(ValueError, match="SHA256"):
    ReferencePolicyPPO(
      algorithm.actor,
      algorithm.critic,
      algorithm.storage,
      reference_coefficient=1,
      reference_checkpoint=str(tmp_path / "reference.pt"),
      reference_sha256="wrong",
      schedule="fixed",
    )
  with pytest.raises(ValueError, match="fixed-rate"):
    ReferencePolicyPPO(
      algorithm.actor,
      algorithm.critic,
      algorithm.storage,
      reference_coefficient=1,
      schedule="adaptive",
    )


def test_explicit_reference_coefficient_transition_preserves_other_contracts(tmp_path):
  algorithm, _ = _algorithm(tmp_path, coefficient=100, resume_coefficient=25)
  state = algorithm.save()
  state["reference_policy"]["coefficient"] = 25
  algorithm.load(state, None, True)
  assert algorithm.save()["reference_policy"]["coefficient"] == 100
  algorithm.load(algorithm.save(), None, True)
  for key, value in (("coefficient", 5), ("sha256", "wrong"), ("mask_key", "wrong")):
    altered = deepcopy(state)
    altered["reference_policy"][key] = value
    with pytest.raises(ValueError, match="contract"):
      algorithm.load(altered, None, True)
  strict, _ = _algorithm(tmp_path, coefficient=100)
  with pytest.raises(ValueError, match="contract"):
    strict.load(state, None, True)
  with pytest.raises(ValueError, match="reference_resume_coefficient"):
    _algorithm(tmp_path, resume_coefficient=float("nan"))


def test_reference_mask_tracks_columns_without_policy_inputs():
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(
        curriculum=True, sub_terrains={"flat": None, "native": None, "narrow": None}
      )
    ),
    terrain_types=torch.tensor([0, 1, 2]),
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(scene=SimpleNamespace(terrain=terrain), device="cpu", num_envs=3),
  )
  cfg = ObservationTermCfg(
    func=reference_terrain_mask, params={"terrain_names": ("flat", "native")}
  )
  term = reference_terrain_mask(cfg, env)
  torch.testing.assert_close(
    term(env, ("flat", "native")), torch.tensor([[1.0], [1.0], [0.0]])
  )
  terrain.terrain_types[:] = torch.tensor([2, 0, 1])
  torch.testing.assert_close(
    term(env, ("flat", "native")), torch.tensor([[0.0], [1.0], [1.0]])
  )
