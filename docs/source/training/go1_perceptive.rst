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

The main task uses a BVH-accelerated pinhole ``RayCastSensor`` and defaults to
4096 environments, including the stair fine-tuning task. Dense ray scans allocate
several buffers per ray; tune the count to GPU memory and measured training
throughput. Play mode defaults to one environment.

.. code-block:: bash

   uv run train Mjlab-Velocity-Rough-Unitree-Go1-Perceptive

The policy uses ``SpatialSoftmaxCNNModel`` for the depth encoder and an MLP critic.
The depth image remains a separate observation group instead of being flattened into
the proprioceptive vector.

Stair fine-tuning
----------------

``Mjlab-Velocity-Stairs-Unitree-Go1-Perceptive`` uses
``GO1_STAIRS_TERRAINS_CFG``: 40% pyramid stairs, 40% inverted pyramid stairs,
10% flat, 5% random rough, and 5% slope terrain. Steps have 30 cm treads and
increase from 2 to 18 cm across 16 curriculum levels. Initial environments
start on levels 0--4 (2--6.3 cm steps). Play mode previews only the two stair
types at the hardest training difficulty: 18 cm steps with 30 cm treads.
Resets can switch stair type but always retain the maximum step height.
This is a training target, not a guarantee of 18 cm traversal success.

The observation and network configuration matches the rough perceptive task.
To continue from an existing run in ``logs/rsl_rl/go1_perceptive_velocity``:

.. code-block:: bash

   uv run train Mjlab-Velocity-Stairs-Unitree-Go1-Perceptive \
     --agent.resume True --agent.load-run <run-directory> \
     --agent.load-checkpoint model_10000.pt

Replace the run directory and checkpoint filename with your existing checkpoint.
Evaluate ascent and descent separately at fixed heights before raising the target.
The original rough task and the general stair preset remain available unchanged.

The stair task keeps its command speed range fixed as terrain difficulty increases.
It prioritizes planar velocity tracking, permits vertical and roll/pitch transients,
and uses weaker nominal-pose and motion regularization than the rough task.
Thigh-ground contacts incur a penalty; falls beyond 70 degrees terminate training
episodes.

Use the headless evaluator to compare checkpoints at identical heights, speeds,
and randomized spawn poses:

.. code-block:: bash

   uv run python -m mjlab.scripts.evaluate_go1_stairs \
     --checkpoints <checkpoint-path> \
     --heights 0.10 0.14 0.18 --speeds 0.3 0.5 0.8 1.0 \
     --trials 16 --output artifacts/go1_stairs_evaluation.json

Passage requires crossing the stair flight with the expected change in height;
falls and lateral exits invalidate that trial. The evaluator reports ascent and
descent separately. A curriculum level is a training state, not a passage metric.

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
