# maps/

| file | what it is |
|---|---|
| `field.pgm` / `field.yaml` | the original 6 x 4 m test arena |
| `field_2027.yaml` | **ABU Robocon 2027 field spec** — the single source of geometry |
| `gen_field_2027.py` | spec → occupancy layers + top-view render |
| `field_2027_{ground,l1,l2}.{pgm,yaml}` | generated, `map_server` format |
| `field_2027_overview.png`, `field_2027_layers.png` | generated, for eyeballing |

```bash
python maps/gen_field_2027.py                        # regenerate everything
python maps/gen_field_2027.py --resolution 0.01      # finer grid
ros2 launch omni_sim_ros sim.launch.py map_yaml:=$PWD/maps/field_2027_ground.yaml
```

## Why three maps

The 2027 field is three storeys (ground, L1 +600, L2 +900). A single 2-D
occupancy grid cannot express that, so the generator emits one grid per
*driving surface*, all sharing the same origin and resolution — a robot's map
is selected by which deck it is standing on, and world coordinates stay
comparable across all three.

- **ground** — TR's world. Occupied: field boundary, the L1 slab wall, the ramp
  and stair flanks, the Mustika Pillar, the centre divider.
- **l1** — BR's world once it is up. Free only on the L1 slab; the Mustika
  pocket is a hole, the L2 slab is a wall, the perimeter barrier closes the
  edge except at the two Transfer Areas.
- **l2** — the 3 x 3 m top deck and the Central Pillar.

Ramps and stairs are marked **occupied** on every layer: they are slopes/steps,
not drivable floor for a wheeled robot, and a level transition is a discrete
event rather than a path through a grid. Model the climb as a scripted
transition that swaps the active map.

## Provenance — read this before trusting a number

`field_2027.yaml` tags every dimension:

- `[R]` — stated in the rulebook text (v1.0, Aug 2026). Solid.
- `[F]` — scaled off Figure 2 / Figure 3. **Estimate.** Building-spot
  positions, ramp and stair placement, retry/transfer/storage/start-zone
  coordinates and the centre-divider gaps are all `[F]`. Replace them when the
  official dimension drawing or CAD is published; nothing downstream hard-codes
  geometry, so editing the YAML and rerunning the generator is the whole job.

Two `[F]` calls worth flagging:

1. **The Mustika pocket.** The 1000 x 1000 shared area around the Mustika
   Pillar is modelled as a notch cut out of the north edge of the L1 slab, with
   the pillar standing on ground level inside it. That reading matches Figure 3
   and makes the pillar reachable by TR (which may never leave the ground), but
   it is inferred, not stated.
2. **Centre-divider gaps.** The fence is broken wherever it would cross a
   shared area or a building spot. A consequence: on the ground layer the red
   and blue halves are physically connected through the Ground Shared Area.
   That is correct — crossing there is a *referee* call (§6.2.2 Zone Violation),
   not a wall. Zone ownership is not encoded in the occupancy grid; keep it in
   the planner.
