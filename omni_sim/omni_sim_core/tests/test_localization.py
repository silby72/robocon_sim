"""Self-localization: the half that corrects the drift.

``plant/odometry.py`` made dead reckoning drift for real. Nothing corrected it
until this, so the thing to pin down is not "is the estimate small" -- on a
run with perfect odometry the filter is *worse*, because it adds sensor noise
to something that had none -- but "does the error stay bounded while
odometry's grows".
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from omni_sim_core.localization import (LikelihoodField, LikelihoodFieldParams,
                                        MCLParams, ParticleFilter)

ROOT = Path(__file__).resolve().parents[2]
FIELD = ROOT / "config" / "field" / "robocon2027.yaml"

pytestmark = pytest.mark.skipif(not FIELD.exists(), reason="repo config/ not available")


@pytest.fixture(scope="module")
def layers():
    from omni_sim_core.field import LayeredField
    return LayeredField.from_yaml(FIELD)


@pytest.fixture(scope="module")
def ground(layers):
    return LikelihoodField(layers.grid("ground", "loc"))


def _scan(grid, pose, n_beams=240, seed=0):
    from omni_sim_core.sensors.lidar import Lidar, LidarParams
    lidar = Lidar(LidarParams(n_beams=n_beams, enable_motion_distortion=False),
                  grid, np.random.default_rng(seed))
    return lidar._measure(0.0, {"pose": np.asarray(pose, dtype=float)})


# --------------------------------------------------------------------------- #
# likelihood field
# --------------------------------------------------------------------------- #
def test_it_is_built_from_the_localization_raster_not_the_nav_one(layers):
    """The two rasters are built the opposite way on purpose. nav is filled,
    so a pose buried inside a wall scores as a perfect match: every beam ends
    on 'occupied'. loc is the 1-cell surface a LiDAR can actually see."""
    loc = layers.grid("ground", "loc")
    nav = layers.grid("ground", "nav")
    from omni_sim_core.env.occupancy_grid import OCCUPIED
    assert (loc.grid == OCCUPIED).sum() < (nav.grid == OCCUPIED).sum() / 10


def test_an_empty_map_is_refused(layers):
    from omni_sim_core.env.occupancy_grid import OccupancyGrid
    empty = OccupancyGrid(np.zeros_like(layers.grid("ground", "loc").grid),
                          layers.grid("ground", "loc").meta)
    with pytest.raises(ValueError, match="no surfaces"):
        LikelihoodField(empty)


@pytest.mark.parametrize("pose", [(1.2, 1.2, 0.0), (1.5, 7.0, 1.2), (2.0, 2.0, -0.6)])
def test_the_true_pose_scores_best(ground, layers, pose):
    """With the sensor and the map describing the same surfaces, the truth
    must beat its neighbours. It did not before the LiDAR was pointed at the
    loc raster: it was reporting the 50 mm field boundary as a wall, which a
    sensor 185 mm up cannot see and the map correctly does not contain."""
    scan = _scan(layers.grid("ground", "loc"), pose)
    p = np.asarray(pose, dtype=float)
    cand = np.array([p, p + [0.2, 0, 0], p + [0, 0.3, 0], p + [0, 0, 0.35],
                     p + [-0.25, 0.25, 0]])
    scores = ground.log_likelihood(cand, scan.angles, scan.ranges)
    assert int(np.argmax(scores)) == 0, f"scores {np.round(scores, 2)}"


def test_beams_that_hit_nothing_are_dropped(ground, layers):
    """An out-of-range beam says only 'nothing along this ray', which is true
    almost everywhere. Scoring it as a maximally bad endpoint would punish
    poses in open space for being in open space."""
    scan = _scan(layers.grid("ground", "loc"), (1.2, 1.2, 0.0))
    pose = np.array([[1.2, 1.2, 0.0]])
    base = ground.log_likelihood(pose, scan.angles, scan.ranges)[0]
    ranges = scan.ranges.copy()
    ranges[np.isfinite(ranges)] = np.inf          # now nothing hit anything
    assert ground.log_likelihood(pose, scan.angles, ranges)[0] == 0.0
    assert base < 0.0


def test_a_tighter_sigma_is_more_discriminating(layers):
    loose = LikelihoodField(layers.grid("ground", "loc"),
                            LikelihoodFieldParams(sigma_hit_m=0.5))
    tight = LikelihoodField(layers.grid("ground", "loc"),
                            LikelihoodFieldParams(sigma_hit_m=0.03))
    scan = _scan(layers.grid("ground", "loc"), (1.2, 1.2, 0.0))
    cand = np.array([[1.2, 1.2, 0.0], [1.5, 1.2, 0.0]])
    gap = lambda f: float(np.diff(f.log_likelihood(cand, scan.angles, scan.ranges))[0])
    assert abs(gap(tight)) > abs(gap(loose))


# --------------------------------------------------------------------------- #
# particle filter
# --------------------------------------------------------------------------- #
def test_it_is_deterministic_given_a_seed(ground):
    def run():
        pf = ParticleFilter(ground, (1.2, 1.2, 0.0),
                            rng=np.random.default_rng(7))
        for _ in range(5):
            pf.predict((0.05, 0.0, 0.01))
        return pf.estimate()
    assert run() == pytest.approx(run())


def test_the_heading_is_averaged_as_a_direction(ground):
    """The arithmetic mean of 179 and -179 degrees is 0, pointing backwards."""
    pf = ParticleFilter(ground, (1.0, 1.0, np.pi),
                        params=MCLParams(n_particles=200, init_yaw_std_rad=0.02),
                        rng=np.random.default_rng(3))
    assert abs(abs(pf.estimate()[2]) - np.pi) < 0.1


def test_prediction_spreads_and_a_scan_pulls_it_back(ground, layers):
    """The two halves, separately: motion adds uncertainty, measurement
    removes it. A filter where the update does not shrink the spread is not
    using its sensor."""
    pf = ParticleFilter(ground, (1.2, 1.2, 0.0),
                        params=MCLParams(n_particles=400),
                        rng=np.random.default_rng(11))
    for _ in range(20):
        pf.predict((0.05, 0.0, 0.0))
    spread_after_motion = pf.spread()
    scan = _scan(layers.grid("ground", "loc"), (2.2, 1.2, 0.0))
    for _ in range(4):
        pf.update(scan.angles, scan.ranges)
    assert pf.spread() < spread_after_motion


def test_it_resamples_only_when_the_set_collapses(ground, layers):
    """Resampling every update throws away diversity for nothing, and is the
    usual cause of converging early onto a wrong pose."""
    pf = ParticleFilter(ground, (1.2, 1.2, 0.0),
                        params=MCLParams(n_particles=400, ess_ratio=0.0),
                        rng=np.random.default_rng(5))
    scan = _scan(layers.grid("ground", "loc"), (1.2, 1.2, 0.0))
    for _ in range(5):
        pf.update(scan.angles, scan.ranges)
    assert pf.n_resamples == 0, "resampled despite an ESS threshold of 0"


def test_a_level_change_swaps_the_map_and_keeps_the_particles(ground, layers):
    """Three rasters share one world frame but describe different floors.
    Scoring an L1 scan against the ground map matches the wrong building --
    but the robot did not teleport, so the particles stay."""
    pf = ParticleFilter(ground, (3.2, 6.0, 0.0), rng=np.random.default_rng(1))
    before = pf.particles.copy()
    pf.set_field(LikelihoodField(layers.grid("l1", "loc")))
    assert pf.particles == pytest.approx(before)
    assert pf.field.grid is not ground.grid


def test_the_estimate_stays_bounded_while_odometry_runs_away(layers):
    """The property that matters. Not 'smaller than odometry today' -- on a
    run with perfect odometry the filter is worse, because it adds sensor
    noise to something that had none -- but that its error does not grow.
    """
    from omni_sim_core.mechanism.robot_config import robot_config_from_dir
    from omni_sim_core.sensors.encoder import Encoder, EncoderParams
    from omni_sim_core.sensors.lidar import Lidar, LidarParams
    from omni_sim_core.plant.odometry import OdometryParams
    from omni_sim_core.simulator import RobotSim

    loc = layers.grid("ground", "loc")
    cfg, _ = robot_config_from_dir(ROOT)
    cfg = replace(cfg, clock=replace(cfg.clock, dt_sim=2e-3, dt_motor=2e-3),
                  odometry=OdometryParams(slip_ratio=1.03))
    sim = RobotSim(cfg, seed=0)
    start = np.array([1.2, 1.2, 0.0])
    sim.reset(pose=start)
    sim.odom_pose = start.copy()

    enc = Encoder(EncoderParams(), np.random.default_rng(0))
    lidar = Lidar(LidarParams(n_beams=240, rate_hz=10.0,
                              enable_motion_distortion=False), loc,
                  np.random.default_rng(2))
    pf = ParticleFilter(LikelihoodField(loc), start,
                        params=MCLParams(n_particles=300),
                        rng=np.random.default_rng(42))

    rpc, prev = enc._rad_per_count, sim.odom_pose.copy()
    dt, odom_err, mcl_err = cfg.clock.dt_sim, [], []
    for _ in range(int(40.0 / dt)):
        t = sim.clock.t
        cmd = np.array([0.0, 0.25, 0.0]) if int(t // 10) % 2 == 0 \
            else np.array([0.25, 0.0, 0.0])
        sim.step(sim._wheel_torque_from_cmd_vel(cmd, kp=0.5))
        sim.update_odometry(wheel_angle=rpc * np.round(sim.wheel_angle_true / rpc))
        scan = lidar.maybe_sample(t, {"pose": sim.body.pose,
                                      "twist": sim.body.twist_world})
        if scan is None:
            continue
        d = sim.odom_pose - prev
        c, s = np.cos(-prev[2]), np.sin(-prev[2])
        pf.predict((c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]))
        prev = sim.odom_pose.copy()
        pf.update(scan.angles, scan.ranges)
        odom_err.append(np.linalg.norm(sim.body.pose[:2] - sim.odom_pose[:2]))
        mcl_err.append(np.linalg.norm(sim.body.pose[:2] - pf.estimate()[:2]))

    half = len(odom_err) // 2
    assert np.mean(odom_err[half:]) > 2 * np.mean(odom_err[:half]), \
        "odometry did not drift; the test cannot show anything"
    assert np.mean(mcl_err[half:]) < np.mean(odom_err[half:]), \
        (f"MCL {np.mean(mcl_err[half:]):.3f} m vs odometry "
         f"{np.mean(odom_err[half:]):.3f} m in the second half")
    assert max(mcl_err) < 0.5, f"the filter diverged (max {max(mcl_err):.2f} m)"
