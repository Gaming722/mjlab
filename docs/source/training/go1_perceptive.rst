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
Yaw tracking uses the rate of horizontal heading, including roll/pitch coupling,
with a tighter tolerance to reduce turning drift during ascent.
Thigh-ground contacts incur a penalty; falls beyond 70 degrees terminate training
episodes.
Commanded planar motion below half the requested speed incurs a directional
deficit cost; standing and pure-yaw commands are exempt. The evaluator also
reports actual forward speed and the fraction of time the trunk touches terrain.

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

For a policy that can already climb stairs but struggles at low forward speeds,
``Mjlab-Velocity-Stairs-LowSpeed-Unitree-Go1-Perceptive`` provides a second-stage
fine-tuning task. It samples 12--18 cm steps across seven levels, allocates 60% of
environments to bottom-spawn ascent, and emphasizes frontal low-speed commands.
It removes random pushes during this practice stage while retaining startup
randomization. Resume a capable checkpoint rather than starting this task from
scratch, and check higher speeds and descent for forgetting:

.. code-block:: bash

   uv run train Mjlab-Velocity-Stairs-LowSpeed-Unitree-Go1-Perceptive \
     --agent.resume True --agent.load-run <run-directory> \
     --agent.load-checkpoint <checkpoint-file> --agent.max-iterations 400 \
     --agent.run-name low_speed_stairs

Balanced terrain locomotion
---------------------------

``Mjlab-Velocity-Robust-Unitree-Go1-Perceptive`` balances flat terrain, ascending
and descending stairs, both slope directions, random rough terrain, and waves.
Stairs retain the 2--18 cm range. It samples forward, reverse, lateral, standing,
and explicit pure-yaw commands; the yaw command range is ``[-1, 1] rad/s``.
Linear commands remain moderate rather than expanding with training time.

Planar tracking is separate from explicit ``lin_vel_z`` and ``ang_vel_xy``
quadratic penalties. Action-rate, action-acceleration, joint-acceleration, torque,
and landing penalties discourage oscillation and hard footfalls. Failed episodes
incur a terminal cost. The curriculum promotes successful tile crossings and
does not promote falls. Gentle random pushes remain enabled during training.

Episodes last up to 30 seconds and successful routes extend 7.2 m, retaining
post-descent transitions across patch boundaries. Vertical velocity has weight -8;
roll/pitch rates have weight -0.2. A separate -2 risk cost applies only when the
front feet bear load, both hind feet are unloaded, and the base pitches nose-down.
Normal diagonal support and nose-up ascent are exempt from this risk cost.

Pure-turn commands occupy 30% of the sampling distribution before standing
overrides. Yaw tracking has weight 2. During low-speed turns on uneven ground
(foot-ground height spread above 5 cm, planar command below 0.4 m/s, and yaw
command above 0.4 rad/s), clearance targets increase from 10 to 15 cm. A -1
support cost discourages unloading an entire front or hind pair in those turns;
diagonal support remains allowed. Flat turns and straight motion retain their
original clearance and support objectives.

.. code-block:: bash

   uv run train Mjlab-Velocity-Robust-Unitree-Go1-Perceptive \
     --agent.resume True --agent.load-run <run-directory> \
     --agent.load-checkpoint <checkpoint-file> --agent.max-iterations 2000 \
     --agent.run-name robust_locomotion

Edge-placement fine-tuning
--------------------------

``Mjlab-Velocity-Edges-Unitree-Go1-Perceptive`` increases stair exposure while
retaining all seven terrain families. Stairs occupy 60% of the sampling weight,
with 27/30/33 cm treads and a minimum 1 m central platform. Integer step counts
may leave a slightly larger platform. Flat terrain occupies 20%, with the rest
distributed over rough, wave, and slope terrain. Commands last 6--10 seconds
and include straight motion, standing, reverse/lateral, and full-range yaw.
The observation/action contract and robot control parameters are unchanged.
The edge cost uses new-contact and loaded-support multipliers of 20 and 1,
respectively, under the -1 reward weight. The Robust task retains its gentler
10 and 0.25 multipliers.

This task is intended to fine-tune a capable checkpoint. It uses a fixed
learning rate of ``5e-5``. Its runner restores parameters and optimizer moments,
then applies the configured fixed rate instead of inheriting the parent's rate.

.. code-block:: bash

   uv run train Mjlab-Velocity-Edges-Unitree-Go1-Perceptive \
     --agent.resume True --agent.load-run <run-directory> \
     --agent.load-checkpoint <checkpoint-file> --agent.max-iterations 600 \
     --env.scene.num-envs 2048 --agent.run-name edge_placement

Use the same task with ``play --viewer viser --checkpoint-file <path>`` to view
the training layout and colored edges. Use the Robust task for comparison on the
original 30 cm treads and larger central platforms. Assess both, together with
continuous varied-terrain trials, before selecting a checkpoint.

For experiments with more precise value inputs, ``mdp.feet_edge_proximity`` can
be appended to the critic's observation terms. Its four per-foot values use the
same edge geometry as the reward. Existing actor observations remain unchanged.
The term must be appended after the original critic inputs when warm-starting.
``mjlab.tasks.velocity.rl.critic_checkpoint.expand_edge_critic_inputs`` returns
a copy of a loaded checkpoint with zero-weight columns for four extra inputs,
zero/unit normalizer defaults, and zero-padded optimizer moments. It preserves
the actor and the existing critic function at initialization, and rejects
ambiguous optimizer shapes. Existing task defaults do not add this observation.

Continuous evaluation
---------------------

Robust training also penalizes new contacts and loaded support within 5 cm of
terrain edges extracted from neighboring ``terrain_scan`` scandots. The scan
spacing is 2.5 cm; the critic
still receives its original 10 cm grid through subsampling. Height jumps must
exceed 3 cm in both neighboring surface tangent planes, excluding continuous
slopes. Edge locations are estimates between rays, with finite grid resolution.
The original foot-clearance scan and actor inputs are unchanged.
The proximity cost decreases linearly from one at an estimated edge to zero at
5 cm. A new contact has multiplier 10 and ongoing loaded support has multiplier
0.25, both under the -1 reward weight. Swing feet without a new contact are
exempt. This encourages safe landing choices without a large persistent incentive
to unload a foot immediately after it has landed.

In Robust and stair play tasks, enable the ``feet_edge`` reward debug overlay to
see ordinary scan points in green, extracted upper edges in orange, loaded feet
near an edge in red, and other feet in blue. The overlay uses the same edge map
and distance threshold as the reward.
Raw ray overlays start unchecked to keep these colors visible; their viewer
checkboxes remain available. Ray visualization does not affect depth observations.

Stair passage is only one diagnostic. Use continuous trials to check falls after
the last step, changes in direction, and vibration on flat and uneven terrain:

.. code-block:: bash

   uv run python -m mjlab.scripts.evaluate_go1_locomotion \
     --checkpoints <checkpoint-path> --trials 8 --seed 43 \
     --output artifacts/go1_locomotion/evaluation.json

The default evaluates all seven training terrain types at four difficulties,
with randomized initial heading and 40-second command sequences. Each original
8 m patch is surrounded by flat runout in a 64 m cell, so a trial cannot be
mislabeled after entering a different terrain. Use ``--tile-spacing 8`` for
continuous mixed-terrain stress tests instead. Failure remains
latched for the entire trial. Report survival together with tracking and traversal:
standing still must not be mistaken for good locomotion. Vibration metrics are
computed for full-trial survivors so early falls cannot make a model look smooth.
The command profiles include pure yaw at both -1 and +1 rad/s.
The maneuver and yaw profiles repeat every 40 seconds for longer stress trials,
for example ``--profiles maneuver yaw --duration 120``.
Per-trial friction and failure positions are recorded. ``--friction 1`` selects
a nominal-friction comparison; omitting it retains startup friction randomization.
Edge reports include scan-based landing/support fractions and an independent
3 cm distance test against the exact square risers of pyramid stairs. The latter
counts only contacts in the stair flight, excluding the central platform and flat
runout. Raw event counts are saved for weighted aggregation; no-contact cases
report null rather than zero. Compare edge rates alongside traversal, stalls,
survival, and base-motion metrics.
Use ``--stair-step-width`` and ``--stair-platform-width`` to check alternate
stair geometry; omitting them retains the task defaults. Exact edge locations
are recomputed for that geometry rather than assuming a 30 cm tread.
For repeated flights and neighboring terrain transitions, combine
``--tile-spacing 8 --terrain-rows 8``. Scan metrics and failure checks follow the
entire route; exact geometric stair counts remain restricted to the original
spawn patch. Loaded-foot slip is reported for full-trial survivors, and sustained
missing ground beneath the base is a failure even if the robot remains upright.

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
