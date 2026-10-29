from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers import ObservationTermCfg
from mjlab.sensor import CameraSensor, ContactSensor, RayCastSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class depth_camera:
  """Rate-limited D435-like optical-Z observation in ``(B, 1, H, W)`` format.

  Ray-cast sensors report distance along each unit ray, whereas a RealSense depth
  frame reports optical-axis Z.  This term performs that conversion and holds the
  last captured frame so a 50 Hz policy can consume a nominal 30 Hz camera stream.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    width = int(cfg.params["width"])
    height = int(cfg.params["height"])
    fovy = float(cfg.params["fovy"])
    policy_hz = int(cfg.params["policy_hz"])
    sensor_hz = int(cfg.params["sensor_hz"])
    if width <= 0 or height <= 0:
      raise ValueError("Depth image dimensions must be positive.")
    if not 0 < sensor_hz <= policy_hz:
      raise ValueError("sensor_hz must be in (0, policy_hz].")

    self.width = width
    self.height = height
    self.near = float(cfg.params["near"])
    self.far = float(cfg.params["far"])
    self.policy_hz = policy_hz
    self.sensor_hz = sensor_hz
    self.noise_std = float(cfg.params.get("noise_std", 0.0))
    self.dropout_prob = float(cfg.params.get("dropout_prob", 0.0))
    self._phase = 0
    self._last_step = -1
    self._initialized = False
    self._has_pending_reset = False
    self._reset_pending = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._frame = torch.full(
      (env.num_envs, 1, height, width),
      self.far,
      device=env.device,
      dtype=torch.float32,
    )

    tan_y = torch.tan(torch.tensor(fovy * torch.pi / 360.0, device=env.device))
    tan_x = tan_y * width / height
    u = (torch.arange(width, device=env.device) + 0.5) / width * 2.0 - 1.0
    v = 1.0 - (torch.arange(height, device=env.device) + 0.5) / height * 2.0
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
    self._ray_to_optical_z = torch.rsqrt(
      1.0 + (grid_u * tan_x).square() + (grid_v * tan_y).square()
    ).view(1, 1, height, width)

  def _read(self, env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    if isinstance(sensor, RayCastSensor):
      distances = sensor.data.distances
      expected = self.height * self.width
      if distances.shape[1] != expected:
        raise ValueError(
          f"RayCastSensor '{sensor_name}' has {distances.shape[1]} rays, "
          f"expected {expected}."
        )
      ranges = distances.view(env.num_envs, 1, self.height, self.width)
      depth = ranges * self._ray_to_optical_z
      depth = torch.where(ranges >= 0.0, depth, self.far)
    elif isinstance(sensor, CameraSensor):
      raw = sensor.data.depth
      if raw is None:
        raise ValueError(f"CameraSensor '{sensor_name}' does not provide depth.")
      depth = raw.permute(0, 3, 1, 2)
      if depth.shape[2:] != (self.height, self.width):
        raise ValueError(
          f"CameraSensor '{sensor_name}' returned {tuple(depth.shape[2:])}, "
          f"expected {(self.height, self.width)}."
        )
      depth = torch.nan_to_num(depth, nan=self.far, posinf=self.far, neginf=self.far)
    else:
      raise TypeError(
        f"depth_camera requires RayCastSensor or CameraSensor, got "
        f"{type(sensor).__name__}."
      )
    return depth.clamp(self.near, self.far)

  def __call__(
    self, env: ManagerBasedRlEnv, sensor_name: str, **_: object
  ) -> torch.Tensor:
    step = int(env.common_step_counter)
    raw: torch.Tensor | None = None
    if not self._initialized or self._has_pending_reset:
      raw = self._read(env, sensor_name)
    if not self._initialized:
      assert raw is not None
      self._frame.copy_(raw)
      self._initialized = True
      self._last_step = step
    elif self._has_pending_reset:
      assert raw is not None
      self._frame[self._reset_pending] = raw[self._reset_pending]
      self._reset_pending.zero_()
      self._has_pending_reset = False

    if step != self._last_step:
      self._last_step = step
      self._phase += self.sensor_hz
      if self._phase >= self.policy_hz:
        self._phase -= self.policy_hz
        self._frame.copy_(self._read(env, sensor_name))

    result = self._frame
    if self.noise_std > 0.0:
      result = result + torch.randn_like(result) * self.noise_std
    if self.dropout_prob > 0.0:
      result = torch.where(
        torch.rand_like(result) < self.dropout_prob,
        self.far,
        result,
      )
    return result.clamp(self.near, self.far)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      self._initialized = False
      self._has_pending_reset = False
      self._reset_pending.zero_()
      self._phase = 0
      self._last_step = -1
    else:
      self._frame[env_ids] = self.far
      self._reset_pending[env_ids] = True
      self._has_pending_reset = True


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
