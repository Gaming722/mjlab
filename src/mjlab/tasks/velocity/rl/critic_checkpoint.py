"""Warm starts for appending privileged edge features to an MLP critic."""

from copy import deepcopy

import torch


def expand_edge_critic_inputs(checkpoint: dict, num_features: int = 4) -> dict:
  """Append zero-weight inputs while preserving the actor and Adam history.

  New observations must be appended after the existing critic observations.
  Normalizer defaults are zero mean and unit variance. Optimizer moments for
  the first critic weight are zero-padded; ambiguous matrix shapes are rejected
  rather than guessing which optimizer parameter belongs to the critic.
  The supplied checkpoint is never mutated.
  """
  if num_features <= 0:
    raise ValueError("The number of additional critic features must be positive")
  result = deepcopy(checkpoint)
  critic = result["critic_state_dict"]
  weight = critic["mlp.0.weight"]
  if weight.ndim != 2:
    raise ValueError("Expected an MLP critic with a matrix first-layer weight")
  old_shape = weight.shape
  critic["mlp.0.weight"] = torch.cat(
    (weight, weight.new_zeros(weight.shape[0], num_features)), dim=1
  )
  normalizer_keys = {
    "obs_normalizer._mean": 0.0,
    "obs_normalizer._var": 1.0,
    "obs_normalizer._std": 1.0,
  }
  for name, default in normalizer_keys.items():
    if name not in critic:
      continue
    value = critic[name]
    if value.shape != (1, old_shape[1]):
      raise ValueError(f"Unexpected critic normalizer shape for {name}")
    critic[name] = torch.cat((value, value.new_full((1, num_features), default)), dim=1)
  optimizer = result.get("optimizer_state_dict", {})
  candidates = [
    state
    for state in optimizer.get("state", {}).values()
    if any(
      isinstance(value, torch.Tensor) and value.shape == old_shape
      for value in state.values()
    )
  ]
  if len(candidates) > 1:
    raise ValueError("Ambiguous optimizer parameter for the first critic weight")
  for state in candidates:
    for name, value in state.items():
      if not isinstance(value, torch.Tensor) or value.ndim == 0:
        continue
      if value.shape != old_shape:
        raise ValueError(f"Unsupported critic optimizer state shape for {name}")
      state[name] = torch.cat(
        (value, value.new_zeros(value.shape[0], num_features)), dim=1
      )
  return result
