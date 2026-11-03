"""Scandot discontinuities and contact-gated foot placement penalties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, GridPatternCfg, RayCastSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def grid_neighbor_pairs(nx: int, ny: int, device: str) -> torch.Tensor:
  """Indices of horizontal and vertical neighbors in an XY-indexed grid."""
  indices = torch.arange(nx * ny, device=device).reshape(ny, nx)
  a = torch.cat((indices[:, :-1].flatten(), indices[:-1].flatten()))
  b = torch.cat((indices[:, 1:].flatten(), indices[1:].flatten()))
  return torch.stack((a, b))


def scan_edge_points(
  points: torch.Tensor,
  normals: torch.Tensor,
  distances: torch.Tensor,
  pairs: torch.Tensor,
  height_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Extract approximate edge positions from neighboring scandots.

  Reject missing/backface hits. A height jump must exceed the threshold in
  both local tangent planes, so a continuous steep slope is not an edge.
  XY is the midpoint of the bracketing rays; visualized Z is the upper surface.
  """
  finite = torch.isfinite(points).all(-1) & torch.isfinite(normals).all(-1)
  valid = (distances >= 0) & (normals[..., 2] > 0.2) & finite
  points = torch.nan_to_num(points)
  normals = torch.nan_to_num(normals)
  a, b = pairs
  p, q = points[:, a], points[:, b]
  delta = q - p
  residual_p = (delta * normals[:, a]).sum(-1).abs()
  residual_q = (delta * normals[:, b]).sum(-1).abs()
  edge = (
    valid[:, a]
    & valid[:, b]
    & (delta[..., 2].abs() > height_threshold)
    & (torch.minimum(residual_p, residual_q) > height_threshold)
  )
  lower = torch.minimum(p[..., 2], q[..., 2])
  upper = torch.maximum(p[..., 2], q[..., 2])
  centers = (p + q) * 0.5
  centers[..., 2] = upper
  return centers, edge, lower, upper


def foot_edge_distances(
  feet: torch.Tensor,
  edges: torch.Tensor,
  valid: torch.Tensor,
  lower: torch.Tensor,
  upper: torch.Tensor,
  height_tolerance: float = 0.08,
) -> torch.Tensor:
  """Nearest edge XY distance, matching either adjacent support elevation."""
  distance = torch.cdist(
    feet[..., :2], edges[..., :2], compute_mode="donot_use_mm_for_euclid_dist"
  )
  foot_z = feet[..., 2, None]
  dz = torch.minimum((foot_z - lower[:, None]).abs(), (foot_z - upper[:, None]).abs())
  eligible = valid[:, None] & (dz < height_tolerance)
  return torch.where(eligible, distance, torch.inf).amin(-1)


class feet_edge:
  """Penalize edge landings and support with a smooth distance cost."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._params = cfg.params
    self._debug_vis_enabled = True
    sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(sensor, RayCastSensor)
    pattern = sensor.cfg.pattern
    if not isinstance(pattern, GridPatternCfg) or sensor.num_frames != 1:
      raise ValueError("feet_edge requires a single-frame grid terrain scan")
    resolution = pattern.resolution
    rx, ry = (
      (resolution, resolution) if isinstance(resolution, (int, float)) else resolution
    )
    nx = round(pattern.size[0] / rx) + 1
    ny = round(pattern.size[1] / ry) + 1
    self._pairs = grid_neighbor_pairs(nx, ny, env.device)
    self.edge_mask = torch.zeros((env.num_envs, 4), dtype=torch.bool, device=env.device)
    self.loaded = torch.zeros_like(self.edge_mask)
    self.edge_distance = torch.full_like(self.edge_mask, torch.inf, dtype=torch.float)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    asset_cfg: SceneEntityCfg,
    margin: float = 0.06,
    height_threshold: float = 0.03,
    force_threshold: float = 5.0,
    landing_weight: float = 5.0,
    support_weight: float = 0.25,
  ) -> torch.Tensor:
    sensor: RayCastSensor = env.scene[height_sensor_name]
    data = sensor.data
    edges, valid, lower, upper = scan_edge_points(
      data.hit_pos_w, data.normals_w, data.distances, self._pairs, height_threshold
    )
    feet = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids]
    self.edge_distance = foot_edge_distances(feet, edges, valid, lower, upper)
    self.edge_mask = self.edge_distance < margin
    contact: ContactSensor = env.scene[sensor_name]
    force = contact.data.force
    assert force is not None
    self.loaded = force.norm(dim=-1) > force_threshold
    landing = contact.compute_first_contact(env.step_dt)
    env.extras["log"]["Metrics/feet_edge_support_fraction"] = (
      self.edge_mask & self.loaded
    ).sum() / self.loaded.sum().clamp_min(1)
    env.extras["log"]["Metrics/feet_edge_landing_fraction"] = (
      self.edge_mask & landing
    ).sum() / landing.sum().clamp_min(1)
    proximity = (1.0 - self.edge_distance / margin).clamp(0.0, 1.0)
    contact_weight = support_weight * self.loaded + landing_weight * landing
    return (proximity * contact_weight).sum(-1)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.edge_mask[env_ids] = False
    self.loaded[env_ids] = False
    self.edge_distance[env_ids] = torch.inf

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled:
      return
    env = self._env
    sensor: RayCastSensor = env.scene[self._params["height_sensor_name"]]
    ids = list(visualizer.get_env_indices(env.num_envs))
    if not ids:
      return
    data = sensor.data
    points = data.hit_pos_w[ids]
    distances = data.distances[ids]
    edges, valid, _, _ = scan_edge_points(
      points,
      data.normals_w[ids],
      distances,
      self._pairs,
      self._params.get("height_threshold", 0.03),
    )
    asset_cfg = self._params["asset_cfg"]
    feet = env.scene[asset_cfg.name].data.site_pos_w[ids][:, asset_cfg.site_ids]
    scan_valid = ((distances >= 0) & (data.normals_w[ids, :, 2] > 0.2)).cpu()
    valid = valid.cpu()
    point_positions = points.cpu().numpy()
    edge_positions = edges.cpu().numpy()
    foot_positions = feet.cpu().numpy()
    risky_feet = (self.edge_mask[ids] & self.loaded[ids]).cpu()
    for k, env_id in enumerate(ids):
      for i in scan_valid[k].nonzero().flatten().tolist():
        visualizer.add_sphere(
          point_positions[k, i],
          0.005,
          (0.1, 0.8, 0.2, 0.6),
          label=f"feet_edge_{env_id}_scan_{i}",
        )
      for i in valid[k].nonzero().flatten().tolist():
        center = edge_positions[k, i].copy()
        center[2] += 0.012
        visualizer.add_sphere(
          center,
          0.012,
          (1.0, 0.5, 0.0, 1.0),
          label=f"feet_edge_{env_id}_edge_{i}",
        )
      for i in range(feet.shape[1]):
        risky = bool(risky_feet[k, i])
        color = (1.0, 0.05, 0.05, 0.9) if risky else (0.0, 0.5, 1.0, 0.8)
        visualizer.add_sphere(
          foot_positions[k, i],
          0.025,
          color,
          label=f"feet_edge_{env_id}_foot_{i}",
        )
