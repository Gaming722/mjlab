# Go1 checkpoint catalog

These weights, their actual saved configurations, and provenance are maintained
in Git. Run `git pull` on `feat/go1-stair-depth-finetune` to obtain them; no ignored
`logs/` or `artifacts/` directory is required to retrieve the archived weights.
Each checkpoint is stored once. `catalog.json` records portable paths and SHA256
hashes for all weights and their environment/agent configurations.

**Current hardware-test candidate: V40 model_20500.pt.** V42 model_22499.pt is
the final newer experiment, not a replacement recommendation. V40 passed native
retention and the six-width stair matrix, including 20/24 hard narrow-stair
stop/restart trials. It failed the 120-second continuous-terrain acceptance test;
physical robot performance is unverified. See the
[selection review](go1_stair_depth_hardware_candidate/latest_training_selection_review.json)
and [deployment guide](go1_stair_depth_hardware_candidate/TEST_GUIDE.md).

| Checkpoint | Role |
| --- | --- |
| [V40 model_20500.pt](go1_stair_depth_retention_parent/model_20500.pt) | Current hardware-test candidate; also V41/V42 parent |
| [V29 model_14500.pt](go1_stair_depth_reference/model_14500.pt) | Verified comparison baseline and frozen reference |
| [V42 model_22499.pt](go1_stair_depth_latest/model_22499.pt) | Final V42 experiment; 2000 new updates from V40; not selected |
| [model_19998.pt](go1_stair_depth_legacy/model_19998.pt) | Historical stairs policy previously deployed to NX |
| [model_9200.pt](go1_torque_smooth_baseline/model_9200.pt) | Starting point before torque/smoothness refinement |
| [V30b model_15000.pt](go1_stair_depth_initial_parent/model_15000.pt) | Initial stair-depth adaptation ancestor |
| [V31 model_15100.pt](go1_stair_depth_balanced/model_15100.pt) | Rehearsal checkpoint used to initialize V33 |
| [V33 model_15200.pt](go1_stair_depth_reference/model_15200.pt) | Balanced-depth ancestor; not selected for deployment |
| [V39 model_16750.pt](go1_stair_depth_stratified_parent/model_16750.pt) | Stratified-depth ancestor used to initialize V40 |
| [V31 model_15399.pt](go1_stair_depth_longstop/model_15399.pt) | Final rehearsal checkpoint; V32 comparison branch |

The recent candidate's parent chain is:

```text
V29 14500 -> V30b 15000 -> V31 15100 -> V33 15200 -> V39 16750
                                                    -> V40 20500 -> V42 22499
```

The full historical lineage is recorded in the bundle JSON files. Only selected
milestones have versioned weights; not every historical ancestor is archived.
Iteration filenames are inherited across resumes and are not counts of new
updates. Use saved counters and lineage records to compare training amounts.

Historical checkpoints can have different observation/configuration contracts.
Use their corresponding saved YAML files and lineage rather than assuming all
weights work with the latest task defaults. The bundles retain actor, critic,
optimizer and normalizer state as training checkpoints, rather than ONNX models.

To verify all cataloged weights and configurations from the repository root:

```sh
uv run python - <<'PY'
import hashlib
import json
from pathlib import Path

catalog = json.loads(Path("checkpoints/catalog.json").read_text())
for entry in catalog["checkpoints"]:
    for field in ("path", "saved_env", "saved_agent"):
        checksum = "sha256" if field == "path" else field + "_sha256"
        path = Path(entry[field])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry[checksum], path
print("Verified all checkpoint and configuration hashes.")
PY
```
