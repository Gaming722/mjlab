"""Scandot edge detection, contact gating, and checkpoint compatibility."""

from unittest.mock import MagicMock

import pytest
import torch

from mjlab.envs.mdp.observations import height_scan
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import GridPatternCfg, RayCastSensor
from mjlab.tasks.velocity.mdp.terrain_edges import (
  feet_edge,
  foot_edge_distances,
  grid_neighbor_pairs,
  scan_edge_points,
)


def _scan(kind: str):
  pattern = GridPatternCfg(size=(0.2, 0.1), resolution=0.05)
  points, _ = pattern.generate_rays(None, "cpu")
  normals = torch.zeros_like(points)
  normals[:, 2] = 1.0
  if kind == "step":
    points[:, 2] = (points[:, 0] > 0.025) * 0.18
  elif kind == "slope":
    points[:, 2] = points[:, 0]
    normals[:, 0] = -1.0
    normals /= normals.norm(dim=-1, keepdim=True)
  points[:, :2] += 1000.0  # Large world coordinates must not corrupt distances.
  sensor = MagicMock(spec=RayCastSensor)
  sensor.cfg = MagicMock()
  sensor.cfg.pattern = pattern
  sensor.num_frames = 1
  sensor.num_rays_per_frame = len(points)
  sensor.data.hit_pos_w = points[None]
  sensor.data.normals_w = normals[None]
  sensor.data.distances = torch.ones(1, len(points))
  return sensor


@pytest.mark.parametrize("kind,expected", [("flat", 0), ("slope", 0), ("step", 3)])
def test_edges_reject_continuous_steep_slopes(kind, expected):
  sensor = _scan(kind)
  edges, valid, low, high = scan_edge_points(
    sensor.data.hit_pos_w,
    sensor.data.normals_w,
    sensor.data.distances,
    grid_neighbor_pairs(5, 3, "cpu"),
    0.03,
  )
  assert int(valid.sum()) == expected
  if expected:
    torch.testing.assert_close(edges[0, valid[0], 0], torch.full((3,), 1000.025))
    torch.testing.assert_close(high[valid], torch.full((3,), 0.18))
    torch.testing.assert_close(low[valid], torch.zeros(3))


def test_edge_distances_match_both_treads_and_ignore_other_elevations():
  feet = torch.tensor(
    [[[1000.035, 1000.0, 0.2], [999.9, 1000.0, 0.02], [1000.025, 1000.0, 0.8]]]
  )
  edges = torch.tensor([[[1000.025, 1000.0, 0.18]]])
  distance = foot_edge_distances(
    feet,
    edges,
    torch.ones(1, 1, dtype=torch.bool),
    torch.zeros(1, 1),
    torch.full((1, 1), 0.18),
  )
  assert distance[0, 0].item() == pytest.approx(0.01, abs=1e-4)
  assert distance[0, 1].item() == pytest.approx(0.125, abs=1e-4)
  assert torch.isinf(distance[0, 2])


def test_edge_cost_requires_contact_and_draws_colored_overlays():
  scan = _scan("step")
  robot = MagicMock()
  robot.data.site_pos_w = torch.tensor([[[1000.035, 1000.0, 0.2]] * 4])
  contact = MagicMock()
  contact.data.force = torch.zeros(1, 4, 3)
  contact.data.force[0, 0, 2] = 20.0
  contact.compute_first_contact.return_value = torch.tensor(
    [[True, False, False, False]]
  )
  env = MagicMock()
  env.device = "cpu"
  env.num_envs = 1
  env.scene.__getitem__.side_effect = {
    "scan": scan,
    "robot": robot,
    "contact": contact,
  }.__getitem__
  cfg = RewardTermCfg(
    func=feet_edge,
    weight=-1.0,
    params={
      "height_sensor_name": "scan",
      "sensor_name": "contact",
      "asset_cfg": SceneEntityCfg("robot", site_ids=[0, 1, 2, 3]),
    },
  )
  term = feet_edge(cfg, env)
  torch.testing.assert_close(term(env, **cfg.params), torch.tensor([1.0]))
  visualizer = MagicMock()
  visualizer.get_env_indices.return_value = [0]
  term.debug_vis(visualizer)
  colors = [call.args[2] for call in visualizer.add_sphere.call_args_list]
  assert (1.0, 0.5, 0.0, 1.0) in colors  # Extracted edges.
  assert (1.0, 0.05, 0.05, 0.9) in colors  # Loaded edge foot.
  assert (0.0, 0.5, 1.0, 0.8) in colors  # Swing feet are not penalized.


def test_dense_scan_preserves_original_critic_samples():
  def observe(resolution, stride):
    sensor = MagicMock(spec=RayCastSensor)
    sensor.cfg = MagicMock()
    sensor.cfg.max_distance = 5.0
    sensor.cfg.pattern = GridPatternCfg(size=(1.6, 1.0), resolution=resolution)
    points, _ = sensor.cfg.pattern.generate_rays(None, "cpu")
    points[:, 2] = points[:, 0].sin() + points[:, 1].cos()
    sensor.num_frames = 1
    sensor.num_rays_per_frame = len(points)
    sensor.data.hit_pos_w = points[None]
    sensor.data.frame_pos_w = torch.tensor([[[0.0, 0.0, 0.4]]])
    sensor.data.distances = torch.ones(1, len(points))
    env = MagicMock()
    env.scene.__getitem__.return_value = sensor
    return height_scan(env, "scan", grid_stride=stride)

  coarse = observe(0.1, 1)
  dense = observe(0.05, 2)
  assert dense.shape == coarse.shape == (1, 187)
  torch.testing.assert_close(dense, coarse)
