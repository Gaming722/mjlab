"""Training-only initial difficulty allocation for fixed terrain replay."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def stratify_narrow_terrain_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  terrain_names: tuple[str, ...],
) -> None:
  """Cover all rows, then emphasize the highest quartile in selected columns.

  Half the environments (at least one per row when possible) cover the full
  range. The remainder cover the highest quartile. Other columns and all terrain
  types remain unchanged. Run at startup, before the first physical reset.
  """
  del env_ids
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    raise ValueError("Stratified difficulty requires a generated terrain grid.")
  generator = terrain.cfg.terrain_generator
  if generator is None or not generator.curriculum:
    raise ValueError("Stratified difficulty requires a curriculum terrain layout.")
  if "terrain_levels" in env.cfg.curriculum:
    raise ValueError("Fixed stratified replay cannot use terrain-level promotion.")
  names = list(generator.sub_terrains)
  if not terrain_names or len(set(terrain_names)) != len(terrain_names):
    raise ValueError("Provide unique, nonempty terrain names.")
  if set(terrain_names) - set(names):
    raise ValueError("Unknown stratified terrain names.")
  num_rows = terrain.terrain_origins.shape[0]
  hard_rows = max(1, math.ceil(num_rows / 4))
  for name in terrain_names:
    ids = (terrain.terrain_types == names.index(name)).nonzero().flatten()
    count = ids.numel()
    if count == 0:
      continue
    full_count = min(count, max(num_rows, math.ceil(count / 2)))
    if full_count < num_rows:
      full = torch.linspace(0, num_rows - 1, full_count, device=ids.device).long()
    else:
      full = torch.arange(full_count, device=ids.device) % num_rows
    hard = (
      torch.arange(count - full_count, device=ids.device) % hard_rows
      + num_rows
      - hard_rows
    )
    terrain.terrain_levels[ids] = torch.cat((full, hard))
    terrain.env_origins[ids] = terrain.terrain_origins[
      terrain.terrain_levels[ids], terrain.terrain_types[ids]
    ]
