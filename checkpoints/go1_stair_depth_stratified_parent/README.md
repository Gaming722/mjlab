# Extended stratified stair replay

V39 `model_16750.pt` is a training parent only, not a selected deployment policy. It passed the native-terrain preservation check and hard 22cm stop/resume cases (2/4 each direction), but failed 22.5cm ascent stop/resume (1/4). Remaining scopes were not run after that conclusive failure.

V40 continues the same Stratified task, reward weights, controller, torque/smoothness constraints, frozen model14500 reference (coefficient25), and learning rate2.5e-6 for6000new updates. The first save is17000 (+251 updates), final22749 (+6000). All original full/native/continuous/holdout gates remain mandatory.

Frozen teacher: `../go1_stair_depth_reference/model_14500.pt`. Full checkpoint ancestry is in `lineage.json`; source training configuration is copied unchanged from V39.
