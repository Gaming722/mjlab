"""Evaluate requested/delivered motor effort alongside continuous locomotion."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.evaluate_go1_locomotion import EvaluationConfig, evaluate
from mjlab.tasks.velocity.config.go1.env_cfgs import configure_go1_motor_monitor
from mjlab.tasks.velocity.mdp.motor_safety import motor_monitor


@dataclass
class MotorEvaluationConfig(EvaluationConfig):
  record_motor_traces: bool = False


class MotorObserver:
  """Collect transitions without changing the policy, rewards, or actuator law."""

  def __init__(self, args: MotorEvaluationConfig):
    self.args = args
    self.samples: list[tuple] = []
    self.metadata: dict = {}
    self._previous_key: tuple | None = None
    self._target: torch.Tensor | None = None
    self._delta: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    checkpoint: Path,
    profile: str,
    elapsed: float,
    active: torch.Tensor,
  ) -> None:
    monitor = motor_monitor(env, "joint_pos")
    count = monitor.acc_sample_count.clamp_min(1)
    if not self.metadata:
      terrain = env.scene.terrain
      assert terrain is not None
      self.metadata = {
        "joint_names": monitor.target_names,
        "effort_limits_nm": monitor.effort_limits.cpu().tolist(),
        "nominal_kp": monitor.nominal_kp.cpu().tolist(),
        "nominal_kd": monitor.nominal_kd.cpu().tolist(),
        "terrain_types": terrain.terrain_types.cpu().tolist(),
        "physics_dt": env.physics_dt,
        "policy_dt": env.step_dt,
        "description": "Guard includes SDK-clipped targets with nominal gains; actual force is delivered simulated effort. Fixed-cap motor, no measured torque-speed calibration.",
      }
      torch.testing.assert_close(
        monitor.acc_sample_count, torch.full_like(count, env.cfg.decimation)
      )
    initial = monitor.initial_guard_ratio
    initial_bad = (
      ~torch.isfinite(initial) | (initial > 1.0 + 1e-5 / monitor.effort_limits)
    ).any(-1)
    key = (str(checkpoint.resolve()), profile)
    target = monitor.guard_position_target
    if key != self._previous_key:
      self._target = target.clone()
      self._delta = torch.zeros_like(target)
    assert self._target is not None and self._delta is not None
    delta = target - self._target
    second = delta - self._delta
    self._target = target.clone()
    self._delta = delta.clone()
    self._previous_key = key
    fields = {
      "guard_peak": monitor.peak_guard_ratio.amax(-1),
      "initial_guard_peak": initial.amax(-1),
      "request_peak": monitor.peak_request_ratio.amax(-1),
      "guard_bad": monitor.guard_exceeded.float(),
      "initial_bad": initial_bad.float(),
      "joint_acc_ms": monitor.joint_acc_sq_sum.sum(-1) / (count * monitor.action_dim),
      "base_lin_acc_ms": monitor.base_lin_acc_sq_sum.sum(-1) / count,
      "base_ang_acc_ms": monitor.base_ang_acc_sq_sum.sum(-1) / count,
      "joint_acc_peak": monitor.peak_joint_acc,
      "base_lin_acc_peak": monitor.peak_base_lin_acc,
      "base_ang_acc_peak": monitor.peak_base_ang_acc,
      "actual_torque_ms": monitor.actual_effort_sq_sum.sum(-1)
      / (count * monitor.action_dim),
      "actual_peak": monitor.peak_actual_ratio.amax(-1),
      "saturation_peak": monitor.peak_saturation_ratio.amax(-1),
      "near_samples": monitor.near_limit_sample_count,
      "saturated_samples": monitor.saturation_sample_count,
      "physics_samples": monitor.acc_sample_count,
      "active": active.float(),
      "target_step_ms": delta.square().mean(-1),
      "target_second_ms": second.square().mean(-1),
      "target_step_peak": delta.abs().amax(-1),
    }
    self.metadata["scalar_names"] = list(fields)
    scalars = torch.stack(list(fields.values()), -1).detach().cpu().numpy()
    joint_peaks = (
      torch.stack((monitor.peak_guard_ratio, monitor.peak_actual_ratio), -1)
      .detach()
      .cpu()
      .numpy()
    )
    self.samples.append((key, elapsed, scalars, joint_peaks))

  def augment(self) -> None:
    data = json.loads(self.args.output.read_text())
    labels = [(t, d) for t in self.args.terrains for d in self.args.difficulties]
    types = np.array(self.metadata["terrain_types"])
    elapsed = np.array([s[1] for s in self.samples])
    scalars = np.stack([s[2] for s in self.samples])
    joint_peaks = np.stack([s[3] for s in self.samples])
    columns = {name: i for i, name in enumerate(self.metadata["scalar_names"])}
    for row in data["results"]:
      key = (row["checkpoint"], row["profile"])
      time_mask = np.array([s[0] == key for s in self.samples])
      times = elapsed[time_mask]
      type_id = labels.index((row["terrain"], row["difficulty"]))
      records = []
      for trial_id, env_id in enumerate(np.flatnonzero(types == type_id)):
        values = scalars[time_mask, env_id]
        valid = values[:, columns["active"]] > 0
        quality = valid & (times >= 1.0)
        samples = values[valid, columns["physics_samples"]].sum()
        bad = values[:, columns["guard_bad"]] > 0
        record = {
          "guard_violation_steps": int((bad & valid).sum()),
          "initial_guard_violation_steps": int(
            (values[valid, columns["initial_bad"]] > 0).sum()
          ),
          "first_guard_violation_s": float(times[bad & valid][0])
          if (bad & valid).any()
          else None,
          "guard_peak_ratio": float(values[valid, columns["guard_peak"]].max()),
          "initial_guard_peak_ratio": float(
            values[valid, columns["initial_guard_peak"]].max()
          ),
          "request_peak_ratio": float(values[valid, columns["request_peak"]].max()),
          "actual_peak_ratio": float(values[valid, columns["actual_peak"]].max()),
          "actual_near90_fraction": float(
            values[valid, columns["near_samples"]].sum() / max(samples, 1)
          ),
          "actual_saturation99_fraction": float(
            values[valid, columns["saturated_samples"]].sum() / max(samples, 1)
          ),
          "guard_peak_per_joint": joint_peaks[time_mask, env_id][valid, :, 0]
          .max(0)
          .tolist(),
          "actual_peak_per_joint": joint_peaks[time_mask, env_id][valid, :, 1]
          .max(0)
          .tolist(),
        }
        weights = values[quality, columns["physics_samples"]]
        for source in ("joint_acc", "base_lin_acc", "base_ang_acc", "actual_torque"):
          record[f"{source}_rms"] = (
            float(
              np.sqrt(
                np.average(values[quality, columns[f"{source}_ms"]], weights=weights)
              )
            )
            if weights.sum() > 0
            else None
          )
        for source in ("target_step", "target_second"):
          record[f"{source}_rms_rad"] = (
            float(np.sqrt(values[quality, columns[f"{source}_ms"]].mean()))
            if quality.any()
            else None
          )
        for source in ("joint_acc", "base_lin_acc", "base_ang_acc", "target_step"):
          record[f"{source}_peak"] = (
            float(values[quality, columns[f"{source}_peak"]].max())
            if quality.any()
            else None
          )
        record["survived"] = row["trial_survived"][trial_id]
        record["guard_free_survived"] = (
          record["survived"] and record["guard_violation_steps"] == 0
        )
        record["crossed"] = row["trial_crossed"][trial_id]
        record["guard_free_crossed_survived"] = (
          record["guard_free_survived"] and record["crossed"]
        )
        records.append(record)
      row["motor"] = {
        "guard_free_survivors": sum(r["guard_free_survived"] for r in records),
        "guard_free_crossed_survivors": sum(
          r["guard_free_crossed_survived"] for r in records
        ),
        "guard_violating_trials": sum(r["guard_violation_steps"] > 0 for r in records),
        "trials": records,
      }
    data["motor_monitor"] = self.metadata
    self.args.output.write_text(json.dumps(data, indent=2) + "\n")
    if self.args.record_motor_traces:
      np.savez_compressed(
        self.args.output.with_suffix(".motor.npz"),
        elapsed=elapsed,
        scalars=scalars,
        joint_peaks=joint_peaks,
        keys=np.array([s[0] for s in self.samples]),
      )


def evaluate_motors(args: MotorEvaluationConfig) -> None:
  observer = MotorObserver(args)

  def factory(*, cfg, device):
    configure_go1_motor_monitor(cfg)
    return ManagerBasedRlEnv(cfg=cfg, device=device)

  evaluate(args, env_factory=factory, step_observer=observer)
  observer.augment()
  data = json.loads(args.output.read_text())
  print(
    json.dumps(
      {
        "motor_results": str(args.output),
        "physical_survivors": sum(r["survivors"] for r in data["results"]),
        "guard_free_survivors": sum(
          r["motor"]["guard_free_survivors"] for r in data["results"]
        ),
        "guard_violating_trials": sum(
          r["motor"]["guard_violating_trials"] for r in data["results"]
        ),
        "guard_free_crossed_survivors": sum(
          r["motor"]["guard_free_crossed_survivors"] for r in data["results"]
        ),
      }
    )
  )


if __name__ == "__main__":
  evaluate_motors(tyro.cli(MotorEvaluationConfig))
