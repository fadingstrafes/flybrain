# FlyBrain Project Handoff

## Goal

Build a local, interactive Drosophila Male CNS connectome simulator/viewer using Janelia `male-cns:v1.0`, with:

- whole-connectome toy neural dynamics
- sensory stimulation experiments
- motor-neuron mapping to fly body parts
- real neuron morphology visualization
- GPU simulation on AMD RX 9070 XT via ROCm/PyTorch
- realtime GPU rendering via OpenGL/ModernGL
- eventual closed-loop virtual fly / arena

This is an experimental connectome-derived simulator, **not a faithful physiological model**. Treat dynamics, neurotransmitter signs, motor kinematics, and behavior as modeling assumptions unless explicitly backed by annotations.

---

## Machine / environment

Project root:

```bash
~/flybrain
```

Python virtual environment:

```bash
cd ~/flybrain
source .venv/bin/activate
```

Linux / Steam Deck environment, 32 GB system RAM.

GPU:

- AMD Radeon RX 9070 XT
- ~15.9 GB VRAM
- PyTorch: `2.12.0+rocm7.14.1`
- ROCm detected: `7.14.60850`
- `torch.cuda.is_available()` returns `True`
- PyTorch uses the `torch.cuda` API on ROCm

GPU sparse benchmark on the real cached connectome:

- GPU: ~51.301 ms / propagation
- speedup vs CPU sparse matvec: ~15.34x
- max CPU/GPU difference in benchmark: 0.0

---

## neuPrint

Server:

```text
https://neuprint.janelia.org
```

Dataset:

```text
male-cns:v1.0
```

Authentication token environment variable:

```bash
NEUPRINT_APPLICATION_CREDENTIALS
```

Do not hard-code, print, or commit the token.

---

## Project layout

```text
~/flybrain/
  .venv/
  src/
  data/raw/
  data/cache/
  runs/
  assets/
```

Important generated caches:

```text
data/cache/neuron_ids.npy
data/cache/connectome_raw.npz
data/cache/nt_sign.npy
data/cache/nt_label.npy
data/cache/metadata.parquet
data/cache/motor_map.csv
```

Raw bulk Male CNS files include:

```text
connectome-weights-male-cns-v1.0-minconf-0.5.feather
body-neurotransmitters-male-cns-v1.0.feather
body-annotations-male-cns-v1.0-minconf-0.5.feather
```

---

## Sparse graph orientation

This is critical.

`build_full.py` constructs:

```python
W[pre_idx, post_idx] = weight
```

So:

```text
W[i, j] = synapse count from neuron i -> neuron j
```

CPU event-driven simulation visits rows of presynaptic neurons.

For a dense/sparse vectorized GPU propagation where `x[pre]` contains presynaptic activity, use the transpose:

```python
post_input = W.T @ x
```

Do not silently reverse this orientation.

---

## Neurotransmitter sign assumptions

Current simplified model:

```text
acetylcholine  +1
GABA           -1
everything else 0
```

Glutamate, dopamine, serotonin, octopamine, unclear transmitters, etc. remain neutral until receptor-specific effects are modeled.

This is explicitly a modeling assumption.

---

## Existing scripts

Important scripts created during development:

```text
src/explore.py
src/identify.py
src/dng15.py
src/neuron.py
src/view.py
src/view_circuit.py
src/sim.py

src/download_full.py
src/inspect_full.py
src/build_full.py
src/sim_full.py
src/sim_full_gpu.py

src/motor_map.py
src/sensory_map.py
src/sim_sensory.py
src/compare_runs.py

src/animate_leg.py
src/animate_fly.py
src/animate_compare_3d.py

src/cache_run_skeletons.py
src/activity_viewer.py
src/skeleton_activity_viewer.py
src/fly_brain_viewer.py
src/fly_brain_viewer_fast.py
src/fly_brain_opengl.py

src/inspect_model.py
```

Prefer inspecting current files before editing; some scripts evolved through multiple iterations.

Keep working CPU versions intact when adding experimental GPU/realtime versions.

---

## CPU whole-CNS simulator behavior

`src/sim_full.py` is the reference implementation.

Important details:

- loads sparse CSR `connectome_raw.npz`
- computes each presynaptic neuron's strongest outgoing anatomical connection (`row_max`)
- outgoing weight normalization is relative to that maximum
- runtime `--min-weight` filters anatomical edges
- propagation is event-driven: only rows belonging to neurons that fired on the previous timestep are traversed
- unknown NT sign (`0`) causes no synaptic effect
- voltage leak is applied each timestep
- external stimulus adds current to selected neurons
- firing threshold resets fired neurons to 0
- avalanche protection can abort when too many neurons fire simultaneously
- saves spike events and counts

Typical arguments:

```text
--stimulate
--target
--steps
--stim-start
--stim-end
--stim-current
--threshold
--leak
--synaptic-gain
--min-weight
--abort-fraction
--max-recorded-events
--output
```

Outputs expected by other tools:

```text
spike_counts.npz
spike_events.npz
top_active.csv
active_motor_neurons.csv
params.json
```

Do not change these formats casually; many viewers depend on them.

---

## Known neurons / sanity checks

Useful known IDs:

```text
12781   DNge104_R
556329  DNge104_L
513052  DNg15_R
10141   DNg15_L
815344  Ti extensor MN_L
```

Known connections:

```text
12781  -> 513052   weight 150
556329 -> 10141    weight 116

513052 -> 800373 IN23B001_L  weight 289
513052 -> 800206 IN23B001_R  weight 188
513052 -> 800002 IN07B010_L  weight 151
513052 -> 815344 Ti extensor MN_L weight 87
```

`815344 Ti extensor MN_L`:

- left front leg
- tibia extensor motor neuron
- `somaNeuromere T1`
- `exitNerve ProLN`
- superclass `vnc_motor`
- mapped action: `extend`

Use these as regression/sanity checks when changing graph propagation or motor mapping.

---

## DNg15 GPU run

A recent GPU run:

```text
runs/dng15_gpu
```

Observed cached activity selection included:

```text
513052 DNg15_R          150 spikes
815344 Ti extensor MN_L   9 spikes
800373 IN23B001_L        41 spikes
800206 IN23B001_R        28 spikes
```

Total recorded spike events were ~469 for that run.

The resulting movement is legitimately subtle because only a small number of mapped motor spikes occur. Do not interpret weak motion as a rendering bug without checking motor spike counts.

---

## Sensory experiments

`sensory_map.py` found ~17,937 sensory neurons.

Broad systems:

```text
other       11157
hind_leg     1521
middle_leg   1402
wing         1282
abdomen      1144
front_leg     992
haltere       439
```

Useful sensory subclasses:

- `leg bristle` = broad touch-like mechanosensory interpretation
- `chordotonal organ` = broad proprioceptive-like interpretation

These are broad interpretation labels, not full physiology.

Strong front-left bristle experiment:

```bash
python -m src.sim_sensory \
  --system front_leg \
  --side L \
  --subclass "leg bristle" \
  --max-neurons 100 \
  --selection strongest \
  --steps 1500 \
  --stim-start 50 \
  --stim-end 150 \
  --stim-current 1.0 \
  --min-weight 1 \
  --synaptic-gain 0.5 \
  --output runs/touch_front_left_strong
```

Chordotonal experiment:

```bash
python -m src.sim_sensory \
  --system front_leg \
  --side L \
  --subclass "chordotonal organ" \
  --max-neurons 100 \
  --selection strongest \
  --steps 1500 \
  --stim-start 50 \
  --stim-end 150 \
  --stim-current 1.0 \
  --min-weight 1 \
  --synaptic-gain 0.5 \
  --output runs/proprio_front_left
```

Comparison previously observed:

```text
Touch run:
  ~467 active motor neurons
  ~9,717 motor spikes

Proprio run:
  ~41 active motor neurons
  ~379 motor spikes
```

These results are consequences of the current toy dynamics, not claims about real fly behavior.

---

## Motor map

Current motor map found ~815 motor neurons.

Broad systems:

```text
leg      325
abdomen  238
other    179
wing      69
haltere    4
```

Leg pair counts:

```text
hind    122
middle  116
front    87
```

Leg actions inferred from annotations include:

```text
generic
flex
extend
abduct
depress
lift
adduct
```

Exit-nerve mapping used:

```text
ProLN   -> front leg
MesoLN  -> middle leg
MetaLN  -> hind leg

ADMN/PDMNa/PDMNp/PDMN/MesoAN -> wing
DMetaN -> haltere
AbN1/AbN2/AbN3/AbN4/AbNT -> abdomen
```

---

## Realtime visualization

### Matplotlib viewers

Working but CPU-heavy:

```text
src/activity_viewer.py
src/skeleton_activity_viewer.py
src/fly_brain_viewer.py
src/fly_brain_viewer_fast.py
```

`fly_brain_viewer_fast.py` improves performance by:

- limiting loaded skeleton count
- hiding inactive skeletons by default
- visual-only motor motion gain
- keeping neural data unchanged

The Matplotlib viewer is now considered a fallback/debug viewer.

### OpenGL / ModernGL viewer

New preferred realtime direction:

```text
src/fly_brain_opengl.py
```

Dependencies:

```bash
pip install moderngl glfw
```

Purpose:

- actual OpenGL render loop
- skeleton coordinates uploaded into GPU VBOs
- procedural fly on left viewport
- reconstructed neurons on right viewport
- neuron activity controls brightness
- mouse orbit / zoom
- title reports FPS and current step

Example:

```bash
python -m src.fly_brain_opengl \
  --run runs/touch_front_left_strong
```

The next rendering goal is to replace the crude procedural fly with a better real-time visual model without returning to slow Matplotlib.

Longer-term goal:

- one realtime engine
- GPU simulation and renderer connected directly
- no mandatory intermediate replay step through `spike_events.npz`
- live stimulus / CNS activity / motor response
- closed-loop arena

---

## Biological fly GLB

There is a biological-looking fly model:

```text
fly.glb
```

It is ~44 MB.

Inspection found:

- 16 geometry objects
- no image/raster textures
- no armature / skin
- no animations
- meshes use per-vertex `COLOR_0` RGBA data
- UV coordinates exist
- original object split is messy and not anatomical
- leg geometry spans multiple mesh chunks and hundreds of disconnected islands
- do NOT assume objects map cleanly to individual legs/body regions

Manual Blender rigging was attempted and intentionally paused because it was too time-consuming.

Do not use `Separate -> By Loose Parts`; this would explode the model into >1000 small objects due to hairs/setae/details.

If revisiting the model, prefer automated or semi-automated rigging, or use it initially as a static visual shell.

---

## Scientific caution

Always distinguish:

1. anatomical connectome data
2. annotations
3. neurotransmitter prediction
4. modeling assumptions
5. emergent output from the toy simulation

Do not phrase simulated behavior as experimentally established fly behavior.

Especially avoid claiming:

- exact real membrane dynamics
- exact receptor effects
- realistic biomechanics
- behavioral validity from the current toy model

---

## Current priorities

Suggested order:

1. Stabilize `fly_brain_opengl.py`
2. Benchmark FPS with 10/20/40 real neuron morphologies
3. Batch real neuron line segments into fewer draw calls
4. Add GPU-rendered UI / timeline / motor bars
5. Connect `sim_full_gpu.py` directly to renderer
6. Add interactive sensory stimulation
7. Add closed-loop arena
8. Revisit realistic fly mesh / rigging later

---

## Working style / safety

- Do not delete working reference scripts when making new experimental versions.
- Prefer `*_gpu.py`, `*_opengl.py`, etc. for major experimental rewrites.
- Verify matrix orientation before modifying propagation.
- Keep output formats compatible with existing viewers where practical.
- Never commit neuPrint credentials.
- Avoid dense NxN matrices; whole CNS graph must remain sparse.
- GPU is AMD ROCm, not NVIDIA CUDA, even though PyTorch APIs say `torch.cuda`.
