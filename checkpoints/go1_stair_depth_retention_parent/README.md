# V40 hardware-test candidate and retention parent

V40 model_20500.pt is the current hardware-test candidate under the user's
revised scope, prioritizing the best available stair policy over universal
terrain acceptance. Native448 and full960 passed; continuous112 failed.
The original strict rejection remains recorded in the historical evaluation
and lineage files. This selection does not claim universal stability or
physical robot validation. See the current
[selection review](../go1_stair_depth_hardware_candidate/latest_training_selection_review.json).

This same file is the V41/V42 training parent. V41 tests reference coefficient100
instead of25, with rewards, constraints, terrain and commands held fixed.
V42 adds hard native-terrain replay with coefficient25. Neither experiment
replaced V40 as the hardware-test candidate. The original evaluation gates
remain unchanged in the archived experiment results.
