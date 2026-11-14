# V40 model20500 hardware test candidate

Selected for prompt hardware testing under the revised user scope. This is not an all-terrain acceptance claim. Native retention and six-width960 trials passed. Hard18cm narrow stop/restart:20/24. Continuous turning on stairs/slopes can still fall; holdout not assessed. Physical robot performance is unverified.

Local upload (new isolated directory; no existing model overwritten):

```sh
scp artifacts/go1_stair_depth/candidates/v40_20500_hardware_candidate/go1-v40-20500-real.tar.gz unitree@192.168.123.25:/home/unitree/
ssh unitree@192.168.123.25
```

On NX, unpack and build using its existing Python3.6 toolchain (NX has no uv):

```sh
mkdir -p /home/unitree/go1-v40-20500-test
cd /home/unitree/go1-v40-20500-test
tar -xzf /home/unitree/go1-v40-20500-real.tar.gz
/usr/bin/python3 scripts/build_native.py --sdk /home/unitree/go1-sim2real/vendor/unitree_legged_sdk --private-libc-headers /home/unitree/go1-sim2real/.build/rs2-2.50.0-py36/libc6-dev-arm64
/usr/bin/python3 scripts/benchmark_perceptive.py --device cpu --iterations 100
bash scripts/play_perceptive_real.sh --config configs/real_go1_low_speed.json
```

The benchmark above is offline and should pass on NX before the motor-enabled entry point. Keep the existing low-level mode and controller arrangement. Entry starts PASSIVE; L1+A stand, L1+X RL, L1+B sit, L1+R1 damping. Sit before Ctrl+C. No robot connection or motor output was performed while preparing this artifact.

Local sim2sim:

```sh
cd ../blind-go1-sim2sim
uv run --no-sync python scripts/play.py --terrain stairs --perceptive-model ../mjlab_fork/artifacts/go1_stair_depth/candidates/v40_20500_hardware_candidate/model_20500.onnx
```
