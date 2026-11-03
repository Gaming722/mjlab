"""Wave origins must not initialize standing feet inside positive crests."""

import mujoco
import numpy as np

from mjlab.terrains.config import wave_terrain


def test_wave_spawn_origin_clears_central_surface():
  cfg = wave_terrain(size=(8.0, 8.0), amplitude_range=(0.2, 0.2), num_waves=4)
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(1.0, spec, np.random.default_rng(43))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  heights = []
  for dx in (-0.4, 0.0, 0.4):
    for dy in (-0.4, 0.0, 0.4):
      point = output.origin.copy()
      point[:2] += (dx, dy)
      point[2] = 2.0
      distance = mujoco.mj_ray(
        model,
        data,
        point,
        np.array([0.0, 0.0, -1.0]),
        None,
        True,
        -1,
        np.zeros(1, dtype=np.int32),
      )
      assert distance >= 0
      heights.append(2.0 - distance)
  assert max(heights) > 0.1
  assert output.origin[2] >= max(heights) - 1e-6
