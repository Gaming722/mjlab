"""Motor safety scoring retains early violations and ignores post-failure motion."""

import json

import numpy as np
import pytest

from mjlab.scripts.evaluate_go1_motor_safety import (
  MotorEvaluationConfig,
  MotorObserver,
)


def test_motor_summary_latches_guard_and_weights_physics_samples(tmp_path):
  checkpoint = tmp_path / "model.pt"
  output = tmp_path / "results.json"
  key = (str(checkpoint.resolve()), "slow")
  output.write_text(
    json.dumps(
      {
        "results": [
          {
            "checkpoint": key[0],
            "profile": key[1],
            "terrain": "flat",
            "difficulty": 1.0,
            "trial_survived": [True, False, True, True],
            "trial_crossed": [True, True, False, True],
          }
        ]
      }
    )
  )
  observer = MotorObserver(
    MotorEvaluationConfig(
      checkpoints=(checkpoint,),
      output=output,
      terrains=("flat",),
      difficulties=(1.0,),
      profiles=("slow",),
      trials=4,
    )
  )
  names = [
    "active",
    "physics_samples",
    "guard_bad",
    "initial_bad",
    "guard_peak",
    "initial_guard_peak",
    "request_peak",
    "actual_peak",
    "near_samples",
    "saturated_samples",
    "joint_acc_ms",
    "base_lin_acc_ms",
    "base_ang_acc_ms",
    "actual_torque_ms",
    "target_step_ms",
    "target_second_ms",
    "joint_acc_peak",
    "base_lin_acc_peak",
    "base_ang_acc_peak",
    "target_step_peak",
  ]
  observer.metadata = {"terrain_types": [0, 0, 0, 0], "scalar_names": names}
  columns = {name: i for i, name in enumerate(names)}
  for step, elapsed in enumerate((0.0, 1.0, 2.0)):
    values = np.ones((4, len(names)))
    values[:, columns["physics_samples"]] = 4 if step < 2 else 2
    values[:, columns["guard_bad"]] = step == 0
    values[:, columns["initial_bad"]] = step == 0
    values[2:, columns["guard_bad"]] = 0
    values[2:, columns["initial_bad"]] = 0
    values[:, columns["joint_acc_ms"]] = 1 if step < 2 else 9
    if step == 2:
      # This robot already fell; huge later values must not affect its scores.
      values[1, :] = 1000
      values[1, columns["active"]] = 0
    observer.samples.append((key, elapsed, values, np.ones((4, 12, 2))))
  observer.augment()
  motor = json.loads(output.read_text())["results"][0]["motor"]
  assert motor["guard_free_survivors"] == 2
  assert motor["guard_free_crossed_survivors"] == 1
  assert motor["guard_violating_trials"] == 2
  first, failed, parked, successful = motor["trials"]
  assert not first["guard_free_crossed_survived"]
  assert not failed["guard_free_crossed_survived"]
  assert not parked["guard_free_crossed_survived"]
  assert successful["guard_free_crossed_survived"]
  assert first["first_guard_violation_s"] == 0
  assert first["guard_violation_steps"] == 1
  assert first["joint_acc_rms"] == pytest.approx(np.sqrt(22 / 6))
  assert first["actual_near90_fraction"] == pytest.approx(3 / 10)
  assert failed["guard_peak_ratio"] == 1
  assert failed["joint_acc_rms"] == 1
  assert failed["actual_near90_fraction"] == pytest.approx(2 / 8)
