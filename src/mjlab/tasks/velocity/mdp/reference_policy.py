"""Training-only replay mask; never included in actor or critic inputs."""

from typing import TYPE_CHECKING

import torch

from mjlab.managers import ObservationTermCfg
from mjlab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["reference_terrain_mask"]


class reference_terrain_mask(ManagerTermBase):
  def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
    super().__init__(env)
    terrain = env.scene.terrain
    generator = terrain.cfg.terrain_generator if terrain is not None else None
    if terrain is None or generator is None or not generator.curriculum:
      raise ValueError("Reference replay requires a curriculum terrain grid.")
    names = list(generator.sub_terrains)
    selected = cfg.params["terrain_names"]
    if not selected or set(selected) - set(names):
      raise ValueError("Reference terrain names must exist in the terrain grid.")
    self.terrain = terrain
    self.columns = torch.tensor([names.index(n) for n in selected], device=env.device)

  def __call__(
    self, env: "ManagerBasedRlEnv", terrain_names: tuple[str, ...]
  ) -> torch.Tensor:
    del env, terrain_names
    return torch.isin(self.terrain.terrain_types, self.columns).float().unsqueeze(-1)
