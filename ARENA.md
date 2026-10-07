# Live fly arena

Run from the project root:

```bash
.venv/bin/python -B -m src.fly_arena
```

The arena runs a live loop: environmental leg/eye sensors → the existing
`male-cns:v1.0` connectome → mapped motor spikes → body movement → sensors.
The window contains an articulated procedural fly, walls and obstacles, a path
trace, six motor bars, contact indicators, live cached neuron morphologies, and
learning statistics. Rendering and neural stepping run separately.

The default room is now **48 × 36 units**, nine times the original floor area.
Use `--arena-width` and `--arena-height` to change it (12–200 each).
Two live eye feeds sample the room and obstacles. A visual braking/steering
assist is enabled by default; press **V** to toggle it. It acts on perceived
clearances and still requires motor activity to power motion. This assist is an
engineered controller, not a reflex discovered by the connectome or learning.

Click **WHOLE CNS** or press **Tab** for a large, interactive anatomical map:
**140,635** positioned cells from the retained live graph. Drag to orbit, scroll
to zoom, click a neuron for its ID/name/class/spike count, and press **Z** to focus
on it. **H** hides inactive cells; **C** resets the camera. Activity flashes yellow,
while resting colors identify annotated cell groups. The sidebar lists recently
active groups, including neurons without positions. The simulation continues
across tabs. Launch directly into this view with `--view brain`.

The map combines soma/root positions with a cached subset of neuron branches,
including ascending/descending arbors connecting the head and ventral nerve cord.
It is not a display of every neurite or synapse. **26,541** retained bodies lack
soma/root coordinates and are omitted from the points. The arena tab keeps the
original small cached morphology display.

The body is a new procedural model. `assets/fly.glb` remains available for later
rigging work and is not animated by this prototype.

Walking now uses world-space planted feet, smooth timed swing arcs, a tripod
phase pattern and two-segment inverse kinematics. Legs tuck in flight. Flight
uses mapped wing motor activity for limited lift, with takeoff, a target altitude,
wing animation and landing. Wingbeat visuals are deliberately slowed for display.

## Controls

| Control | Action |
| --- | --- |
| Space | Pause/resume simulation and learning |
| R | Reset body, neural state and current recording; keep learned policy |
| E | Toggle exploratory sensory drive; environmental sensing remains active |
| 1 / 2 | Brief touch stimulus to left/right leg sensory populations |
| C | Switch between following the fly and viewing the whole arena |
| F | Request takeoff/landing manually; holds flight mode until another request |
| G | Return flight-mode choices to the learning controller |
| V | Toggle visual avoidance assist |
| Tab / sidebar tabs | Switch between arena and whole-CNS map |
| H (CNS view) | Show active neurons only / all neurons |
| Click / Z (CNS view) | Select neuron / focus camera on selection |
| C (CNS view) | Reset anatomical camera |
| Mouse drag | Orbit |
| Scroll | Zoom |
| Esc / close window | Save policy and run outputs, then quit |

Reset starts a new episode and discards the current episode's unsaved spike and
trajectory recording. Close the window first when you want to preserve that run.

## What learns and what persists

A small tabular Q-learning controller chooses among eight actions: balanced,
left bias, right bias, front legs, hind legs, rest, takeoff, and land. Each action
is selected for half a simulated second; takeoff/land requests persist until
changed. Its 256 states encode left/right/front contact, low speed, being airborne,
and nearby visual hazards on the left, ahead and right.
It earns reward for newly visited tiles and distance, with penalties
for repeated contact, being stuck and facing a nearby hazard. It occasionally
tries random actions.

The controller **does not command body speed or modify anatomical connections**.
Its stimulation must pass through the sparse neural simulation to generate motor
output. This is engineered reinforcement learning around the connectome, not a
model of biological synaptic plasticity, and improved navigation is not yet
established by a controlled evaluation.

- Default memory: `data/cache/arena_policy_v1.json`.
- The next launch loads those action values and continues learning.
- Existing version-1 walking and version-2 flight policies migrate to version 3:
  all old values/counters survive; new visual states start untrained. The default
  filename stays unchanged so existing memory is found automatically.
- The policy saves every 100 learning updates (about 50 simulated seconds) and
  on normal exit. An abrupt crash can lose updates since the last save.
- `R` preserves learned action values but resets the episode's visited tiles,
  reward counter, neural state and body position.
- Each exported run includes its final `policy.json` snapshot and telemetry.
- Memory consists of learned action values and lifetime training counters, not
  a persistent spatial map or a recording of every past experience.
- Concurrent training instances should use separate `--policy` paths. The default
  file is shared and concurrent writers do not merge their learning.

Use a separate memory for an independent experiment:

```bash
.venv/bin/python -B -m src.fly_arena --policy data/cache/arena_policy_experiment_b.json
```

Use learned action values without changing them:

```bash
.venv/bin/python -B -m src.fly_arena --evaluate
```

Use fixed exploratory stimulation without the learning controller:

```bash
.venv/bin/python -B -m src.fly_arena --no-learning
```

## Data and assumptions

The source graph is unchanged. A separate derived cache at
`data/cache/live_brain_v1/` retains **167,176** bodies and **25,638,687** anatomical
edges before runtime filtering. It contains every body with a nonzero modeled
transmitter sign plus every graph body with a superclass annotation. The cache
is rebuilt when source file sizes or modification times change.

Omitted bodies all have zero outgoing effect under the present sign model, so
they cannot feed back into retained voltages. Original full-graph outgoing maxima
are preserved before projection and filtering. Therefore the projection preserves
retained dynamics for matching input, apart from floating-point summation details.
It does not preserve counts or monitoring for omitted bodies. The live activity
guard uses 20% of the retained population, rather than 20% of all source body IDs.

Propagation uses sparse `W.T @ activity`. ACh is +1, GABA −1, other transmitters 0.
The live default has min weight 5, gain .35, leak .92 and threshold 1.0. The neural
clock defaults to 60 steps per arena second; this is an imposed time mapping.

Touch probes and stride-phase feedback stimulate annotated leg bristle and
chordotonal populations. Exploratory drive, sensor currents, motor decoding,
differential steering, gait phase, collision geometry and all length/time scales
are modeling assumptions. The fly can encounter walls or become stuck. The
motion is not validated fly behavior, and the body is not a physics-based muscle
or joint model.

`arena_vision.py` raycasts two 90 × 30 RGB/depth views at about 20 Hz, with wide
overlapping fields of view, floor/wall contrast and obstacle silhouettes. Nine
sectors per eye combine proximity, closing motion and contrast. `visual_input.py`
maps these to 144 annotated, sign-compatible T4/T5 neurons in the local Male CNS
data. Spatial retinal assignments are absent from these annotations: the sector
assignment is synthetic and the proximity estimate uses scene depth. This is
neither measured fly retinotopy nor a biological model of compound-eye optics.
The anatomical connectivity and transmitter signs remain unchanged.

Use `--no-avoidance` to retain eye input without steering assistance, or
`--no-vision` to disable both. The assist can reduce collisions but does not
establish that the learned policy has acquired obstacle avoidance.

A flight request stimulates annotated wing-bristle populations. Left/right
sensory bias can influence wing output. The body receives lift only when mapped
wing motors fire; a request by itself cannot lift a silent fly. The flight model
uses toy gravity and vertical damping, aims for 1.8 units of altitude and clears
obstacles above .9 units. Walls bound horizontal travel in flight too. It does not
simulate aerodynamics, realistic wingbeat frequency or detailed landing contact.

Start a manually requested flight:

```bash
.venv/bin/python -B -m src.fly_arena --takeoff
```

## Recording and reproducible checks

Each launch exports to a new timestamped `runs/arena_*` directory by default:

- `spike_events.npz`, `spike_counts.npz`, `params.json`
- `top_active.csv`, `active_motor_neurons.csv`
- `trajectory.csv` with body pose, altitude, vertical speed, wing/leg motor levels,
  contact input, selected action, reward, coverage, visual clearances, assist
  activation and cumulative contact steps
- `visual_input_neurons.csv` identifying the stimulated visual relays when enabled
- `policy.json` when learning/evaluation is enabled

The event log caps at two million spikes; counts continue and truncation is
recorded explicitly. Counts cover the retained live graph, not all source bodies.
The learning policy still saves when event recording reaches its cap.

Run 60 simulated seconds without opening a window:

```bash
.venv/bin/python -B -m src.fly_arena --headless-steps 3600 \
  --policy runs/my_learning_experiment/policy_memory.json \
  --output runs/my_learning_experiment
```

Capture a short visual check:

```bash
.venv/bin/python -B -m src.fly_arena --frames 360 \
  --screenshot runs/arena_preview.png
```

Tests:

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

Validation: 37 tests passed, including vision geometry, visual relay selection,
collision-assist behavior, CNS selection/rendering and memory migration, plus
projection agreement on an
asymmetric signed circuit, collisions, no movement with zero motor drive, neural
reset, policy persistence, policy migration, frozen evaluation, stance-foot
planting, segment lengths and wing-dependent takeoff/landing. A real-data 3,600-step learning
run completed without the activity guard, visited 89 tiles and made 119 updates;
a second process resumed it to 138 updates. These are smoke-test observations,
not evidence that the learned controller is better than a fixed baseline.

A later real-data learning test selected takeoff and landing autonomously and
reached approximately 1.85 altitude units. Its negative total reward also showed
that stuck recovery/navigation still need work. A separate manual-flight test
verified takeoff, altitude control and return to the floor without the activity
guard. The event cap can truncate long flight logs; always check `params.json`.

With the larger room and eyes enabled, paired 60-second real-connectome runs
using fixed exploratory stimulation produced 120 contact steps without visual
assistance and zero with it. The assisted run travelled 63.44 units and recorded
49,788 visual-relay spikes. Both completed without an activity-guard abort.
These single-condition results demonstrate the assist, not learned navigation
superiority; other starts, flight modes and policies still need evaluation.

Next development targets: evaluate policies over multiple seeds and episodes,
improve credit assignment and stuck recovery, add richer sensors/tasks, then
replace the kinematic body with articulated contact physics and a rigged fly mesh.

## Paired learning evaluation

Run an independent experiment with fresh policies:

```bash
.venv/bin/python -B -m src.evaluate_arena --device cuda \
  --seeds 11 22 33 --train-episodes 2 --eval-episodes 2 --steps 1200
```

Each seed trains its own policy, then compares the frozen policy with fixed
exploratory stimulation and an untrained eight-action policy. Each comparison
uses identical obstacle-free starting positions, headings and action RNG seeds.
Training and evaluation starts use separate random streams. The room and eye
inputs match the live arena, and the neural clock is 60 Hz. By default, the
experiment runs separately with visual assistance off and on; `--assist off`
or `--assist on` limits it to one condition. `--arena-width` and `--arena-height`
set the room dimensions.

The evaluator reuses the sparse live brain, resetting neural, body, sensory,
flight and learning episode state between trials. It verifies frozen evaluation
values and counters. Existing interactive policy files are never loaded or
written. Each experiment requires a new output directory under `runs/` (or a new
`--output` path).

Outputs are `config.json` with the protocol and graph manifest fingerprint,
`episodes.jsonl` with individual results, separate policy snapshots for each
seed/assist condition, and `summary.json` with descriptive means and paired
trained-minus-baseline differences. Coverage is counted independently for all
controllers, including the fixed baseline. Contacts, assistance, low speed and
flight are measured in simulation steps. This metrics-only experiment does not
export replay recordings; use `src.fly_arena` for compatible full run exports.

Aborted episodes remain in the individual results and completion counts, are
excluded from paired comparisons, and cause exit status 2. Episodes within a
seed share a trained policy, so pooled episode means are not independent-seed
significance tests. Fixed stimulation never requests flight; the untrained
controller supplies the comparison with the same eight available actions.
The existing learner updates every half second and does not credit the final
interval at episode termination. Short runs validate the experiment machinery;
they cannot establish navigation superiority or biological validity.

## Draw and play media into the CNS

```bash
.venv/bin/python -B -m src.fly_arena --view brain --media /path/to/video.mp4
```

Press **O** to choose a local video/song, or drop a file into the window. Video
frames light up the neuron map; audio-only files generate a scrolling logarithmic
spectrogram. Audio plays through `ffplay`; `--mute-media` suppresses it. FFmpeg
decodes frames on a separate thread. The system file picker uses Zenity/KDialog.
Only local files are accepted; online links are not downloaded by the viewer.

**This is real input to the toy neural simulation.** Visible soma/root positions
sample the image at their screen coordinates. Bright pixels and painted strokes
deliver external current to those neurons, then the existing sparse `W.T`
propagation and motor decoder produce the response. The image itself is also
shown as a stimulus preview on the points/branches. Yellow/orange highlights and
recorded spike counts represent simulation activity. **H** filters the points
and arbors to actual recent activity. The live fly inset shows motor response;
**U** toggles it (shown when the map viewport is at least 850 pixels wide).

The mapping is a direct spatial stimulus, not modeled sight/hearing or retinal
anatomy. Fibers use real cached segments in the soma map's coordinate system;
they do not create artificial edges. Current sampling uses soma/root points,
not every location along an arbor. Neurons overlapping in the camera projection
can all receive input. Changing the view/region changes which neurons are hit.

| Key | Action |
| --- | --- |
| D | Toggle drawing / orbit mode |
| Left drag in drawing mode | Paint a stimulus |
| Shift + left drag | Erase |
| T / [ / ] | Cycle ink color / decrease / increase brush size |
| Ctrl+Z / X | Undo / clear drawing |
| O / file drop | Open local media |
| K / J / Backspace | Pause media / replay / stop and clear media |
| Y | Switch neuron-color preview / translucent video overlay |
| I | Enable/disable current injection from media and ink |
| − / = | Decrease/increase current strength by 0.1 |
| L | Show/hide cached neuron branches |
| U | Show/hide live fly preview |

Default current is `1.1 × brightness`, with a maximum of 2,048 targets per
neural step. Larger illuminated populations are visited in round-robin order;
the whole image is not injected simultaneously. Use `--art-current` (0–3) and
`--art-limit` (1–8,192) to change these. The status strip shows current targets,
total mapped targets, strength and simulation pause/halt state. The existing
activity guard remains active. `--no-art-stimulation` keeps playback visual only.

Ink remains fixed to the screen while you orbit the anatomy. It continues to
stimulate until erased or input is disabled. Pausing/ending media stops media
current but preserves the final picture; painted input remains. Switching to
the arena tab or minimizing the viewer clears the active mapped input. Media
uses wall time, while neural stepping can fall behind under load; playback and
stimulation are interactive, not a frame-exact experimental replay.

Drawings save to the run's `brain_drawing.png`; reload with `--drawing PATH`.
Exports retain the usual spike/trajectory files and add `art_input_neurons.csv`
with per-neuron input-step counts plus input configuration/totals in `params.json`.
These totals do not store the complete frame-by-frame current schedule. Use a
separate policy or `--no-learning` when testing media without training the
interactive controller on those interventions.

The current cache supplies 71 arbors (424,351 segments), including 15 ascending
or descending neurons. These make the head/cord connection visible; the earlier
apparent separation was caused by displaying only soma/root points.
