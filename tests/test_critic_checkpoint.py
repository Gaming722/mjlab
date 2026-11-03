"""A larger privileged critic must preserve its parent and optimizer history."""

from itertools import chain

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab.tasks.velocity.rl.critic_checkpoint import expand_edge_critic_inputs


def _critic(inputs):
  return MLPModel(
    obs=TensorDict({"critic": torch.zeros(2, inputs)}, batch_size=[2]),
    obs_groups={"critic": ["critic"]},
    obs_set="critic",
    output_dim=1,
    hidden_dims=[16, 8],
    activation="elu",
    obs_normalization=True,
  )


def test_edge_critic_expansion_preserves_values_actor_and_adam_moments():
  torch.manual_seed(43)
  actor = torch.nn.Linear(3, 2)
  critic = _critic(7)
  inputs = torch.randn(16, 7) * 2 + 1
  observations = TensorDict({"critic": inputs}, batch_size=[16])
  critic.update_normalization(observations)
  optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()))
  (
    critic(observations).square().mean() + actor(torch.randn(16, 3)).square().mean()
  ).backward()
  optimizer.step()
  optimizer.zero_grad()
  checkpoint: dict = {
    "actor_state_dict": actor.state_dict(),
    "critic_state_dict": critic.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "iter": 123,
    "infos": {"env_state": {"common_step_counter": 456}},
  }
  expanded = expand_edge_critic_inputs(checkpoint)
  assert checkpoint["critic_state_dict"]["mlp.0.weight"].shape == (16, 7)
  assert expanded["infos"] == checkpoint["infos"]
  assert expanded["iter"] == 123
  for name, value in actor.state_dict().items():
    torch.testing.assert_close(
      expanded["actor_state_dict"][name], value, rtol=0, atol=0
    )
  wider = _critic(11)
  wider.load_state_dict(expanded["critic_state_dict"])
  critic.eval()
  wider.eval()
  extra = torch.cat((inputs, torch.randn(16, 4)), dim=-1)
  wider_observations = TensorDict({"critic": extra}, batch_size=[16])
  torch.testing.assert_close(wider(wider_observations), critic(observations))
  new_optimizer = torch.optim.Adam(chain(actor.parameters(), wider.parameters()))
  new_optimizer.load_state_dict(expanded["optimizer_state_dict"])
  old_parameter = dict(critic.named_parameters())["mlp.0.weight"]
  new_parameter = dict(wider.named_parameters())["mlp.0.weight"]
  for moment in ("exp_avg", "exp_avg_sq"):
    old = optimizer.state[old_parameter][moment]
    new = new_optimizer.state[new_parameter][moment]
    torch.testing.assert_close(new[:, :7], old, rtol=0, atol=0)
    assert torch.count_nonzero(new[:, 7:]) == 0
  for old, new in zip(
    chain(actor.parameters(), critic.parameters()),
    chain(actor.parameters(), wider.parameters()),
    strict=True,
  ):
    if old is old_parameter:
      continue
    for name, value in optimizer.state[old].items():
      torch.testing.assert_close(new_optimizer.state[new][name], value, rtol=0, atol=0)
  wider(wider_observations).square().mean().backward()
  new_optimizer.step()  # Restored moments must support a real training update.


def test_edge_critic_expansion_rejects_ambiguous_optimizer_states():
  checkpoint = {
    "critic_state_dict": {"mlp.0.weight": torch.ones(2, 3)},
    "optimizer_state_dict": {
      "state": {
        0: {"exp_avg": torch.ones(2, 3)},
        1: {"exp_avg": torch.ones(2, 3)},
      }
    },
  }
  with pytest.raises(ValueError, match="Ambiguous optimizer"):
    expand_edge_critic_inputs(checkpoint)
