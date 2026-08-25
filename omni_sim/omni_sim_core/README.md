# omni_sim_core

ROS-independent core of the omni-directional robot simulator. Pure
`numpy`/`scipy`; imports **no** `rclpy`, so it runs under `pytest` and parameter
sweeps with no ROS 2 installed.

## Install / test

```bash
pip install -e omni_sim_core[test,viz]
pytest omni_sim_core/tests -q
# or without installing:
PYTHONPATH=omni_sim_core/src pytest omni_sim_core/tests -q
```

## Layers

```
body (true)  <- jacobian (static) <- motor (true)     numeric integration (RK4)
nominal model                                          what the controller believes
disturbance / sensors / env                            measurement + world
```

The single invariant that everything else serves: **the true plant and the
nominal model are configured independently** (`plant/motor.py` vs
`plant/nominal.py`, `plant_true.yaml` vs `plant_nominal.yaml`). The DOB exists to
reconstruct exactly the difference between them.

## Headless run

```bash
PYTHONPATH=omni_sim_core/src python -m omni_sim_core.run \
    --scenario config/scenarios/dob_step_load.yaml
```
