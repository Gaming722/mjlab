Go1 perceptive locomotion
========================

The Go1 perceptive task trains an asymmetric actor-critic policy from D435-like
depth and proprioception.  Its actor receives a 45-value proprioceptive vector and
an ``(1, 45, 80)`` depth image.  The critic retains privileged base linear velocity,
terrain height, and contact observations.

The deployment contract assumes a depth-only RealSense D435 stream at
``640x360 @ 30 Hz`` over USB 2.0, downsampled by eight to ``80x45``.  The simulated
depth is optical-axis Z, clipped to ``[0.1, 2.0]`` metres; misses and invalid pixels
use the far value.  The 50 Hz policy holds the latest frame according to a 30 Hz
capture schedule.  Training additionally uses small depth noise, sparse invalid
pixels, 0--1 control-step latency, and nominal mount perturbations.

The nominal mount is copied from the reference D435 installation: the bottom screw
is at ``(0.26, 0.0, 0.12)`` metres in the trunk frame and pitched down by 30 degrees.
After composing the manufacturer screw-to-depth-origin transform, the optical origin
is ``(0.27542987, 0.0175, 0.12552532)`` metres.  Measure the real mounting transform
and read the active stream intrinsics before deployment; these nominal values are not
a substitute for calibration.

Training
--------

The main task uses a BVH-accelerated pinhole ``RayCastSensor``.  Its default is
conservatively capped at 1024 environments because dense ray scans allocate several
buffers per ray.  Benchmark the target GPU before increasing the count.

.. code-block:: bash

   uv run train Mjlab-Velocity-Rough-Unitree-Go1-Perceptive

The policy uses ``SpatialSoftmaxCNNModel`` for the depth encoder and an MLP critic.
The depth image remains a separate observation group instead of being flattened into
the proprioceptive vector.

Rendered-depth validation
-------------------------

The camera task renders the same nominal projection through ``CameraSensor`` and is
limited to 16 environments.  Use it for small-scale projection, occlusion, and policy
cross-checks rather than primary training.

.. code-block:: bash

   uv run play Mjlab-Velocity-Rough-Unitree-Go1-Perceptive-Camera-Play \
     --agent zero --viewer viser

Neither backend models the D435 stereo matcher by itself.  Real-camera measurements
should eventually replace the initial noise and invalid-pixel settings.
