# FlyBrain development roadmap

Updated 2026-10-07. This is a plan, not a list of completed features.

## Current baseline

Minecraft 26.3 / Fabric supports an independent player-like fly, inventory and
primitive crafting/block interactions, standing/crawling/flight, observer camera,
starter food and renewable berries. Paired RGB cameras stimulate mapped visual
cells; a live dashboard displays the eye inputs and neural activity. A linear
Q readout persists across deaths and sessions around a fixed sparse connectome.

The implementation and integration checks work. Reliable autonomous foraging,
useful construction, visual object recognition and improved survival across
generations remain unproven. More spikes or a rising training reward alone will
not establish those behaviors. Memory persistence does not guarantee improvement.

## 1. Measure survival before changing the learner

Build an isolated Minecraft evaluation harness with reproducible world seeds,
spawn positions, time/weather, inventory and resource layouts. Record the mod,
policy and observation versions, random seeds and episode termination reasons.
Keep training worlds and policies separate from evaluation copies and user saves.

Compare untrained, trained/frozen and random-valid-action policies using the same
starts and at least ten paired seeds initially. Include food scarcity, nearby
hazards and a night-time shelter task. Log survival ticks, food consumed and
harvested, health loss by source, deaths, distance explored, stuck duration,
resource inventory, usable shelter occupancy and each reward component. Include
aborted/failed trials explicitly instead of silently excluding them.

**Done when:** a single documented command produces a machine-readable report
with per-episode results, paired differences and uncertainty, without altering
the user's checkpoint. Capture the current baseline even if it performs poorly.

Start in `src/minecraft_service.py`, `minecraft_brain.py`, `minecraft_learning.py`
and the Fabric test harness. Reuse ideas from `src/evaluate_arena.py`, but do not
treat arena scores as Minecraft survival evidence. Existing `--evaluate` freezes
learning; deterministic Minecraft episode setup still needs implementation.

## 2. Establish useful visual processing

Add frame freshness, capture/bridge/neural timings, direct versus downstream
visual firing, and replayable bounded eye recordings to diagnostics. Keep raw
image recording opt-in to avoid unbounded logs. Exercise turning, occlusion,
lighting changes, unloading and camera restoration in repeatable scenes.

Run paired trials with live eyes, black eyes, shuffled eyes and disabled supplied
food cues. Check whether different images change actions and task outcomes while
holding nonvisual inputs and policy seeds constant. The current 32 visual pools
are coarse; only then compare spatial/motion-preserving readouts and temporal
memory. Keep the source of every feature visible: RGB-derived, depth-derived or
explicit game metadata. Do not silently substitute an object detector for vision.

**Done when:** capture failures are visible, camera restoration passes, image-only
inputs reproducibly reach downstream cells, and the report quantifies whether
vision contributes to obstacle avoidance or food acquisition beyond other cues.
If live eyes do not help, document that result before increasing model complexity.

## 3. Learn sustainable foraging

Create a curriculum of locating a dropped edible item, eating from inventory,
approaching and harvesting ripe berries, and returning after regrowth. Progress
from starter bread to independent food acquisition. Test food on both sides,
occlusion, distance and distractors; use held-out layouts for evaluation.

Inspect reward credit across look/move/select/use sequences. Add short temporal
context or learned action sequences only if baseline failures justify them.
Preserve atomic controls; label any engineered assistance and evaluate it
separately. Check for pickup/drop reward loops, unsafe berry contact, excessive
flight and perpetual consumption/recovery cycles.

**Done when:** a frozen trained policy acquires and eats renewable food after
starter supplies run out on unseen layouts, with improved food balance/survival
over paired baselines. Food category metadata must remain explicitly documented
until an independently evaluated visual association replaces it.

## 4. Progress from placement to useful shelter

First benchmark placing a supplied block, gathering material, and completing
short inventory/crafting sequences. Then train construction and selection of
safe cover using outcomes rather than a hard-coded house blueprint. Compare
existing natural shelter with constructed shelter.

Improve shelter assessment for accessible entrances, connected interior space,
hazards, light and actual protection during night/rain/threat trials. Keep its
engineered definition inspectable. Penalize self-entrapment and verify that a
fly can leave the shelter to forage. Never award success solely for block count.

**Done when:** the fly builds or chooses usable protection with its own collected
materials, can enter/exit, and survives better in held-out threat trials. Scripted
game-test construction remains a capability test, not learned behavior evidence.

## 5. Improve biological provenance and interpretability

Version the facet-to-cell mapping and display annotation/source confidence.
Research retinotopic assignments and retinal transmission before replacing the
current synthetic parallel photoreceptor/L2/L3 stimulation. Preserve the existing
CPU reference and default transmitter signs; any proposed alternative model
needs a separate, explicit experiment and sparse validation.

Investigate adult nociceptive and defensive pathways using evidence that maps
identities into this dataset. A missing name match does not prove absence.
Keep actual transmitter-labeled firing, artificial reward and game-derived
pain/stress estimates distinct. No subjective mood decoder is currently validated.

**Done when:** each claimed circuit mapping has an evidence trail and controlled
stimulation checks; unsupported labels remain marked as modeling assumptions.

## 6. Performance, compatibility and release quality

Measure frame time, GPU memory, eye capture cost and decision latency with and
without the dashboard. Test configured shaders, dimensions and chunk transitions;
currently only default test settings with modpack jars have passed. Make camera
failure recoverable and surface a clear stale-input state. Add reproducible
dependency setup and release artifacts after profiling and compatibility checks.

**Done when:** documented hardware/settings have measured overhead and passing
camera/model tests, and a fresh checkout can reproduce the documented setup.
Dedicated-server vision is a separate design task; the current cameras require
a rendering client and client-loaded terrain.

## Rules for the next session

- Start with milestone 1; do not reset the user's learning to obtain a baseline.
- Use temporary policies and isolated worlds for tests. Back up and migrate any
  new checkpoint schema, including replay features and action order.
- Preserve sparse `W[pre, post]` / `W.T @ activity` math and `src/sim_full.py`.
- Keep learning around the connectome distinct from synaptic plasticity.
- Report failed trials and compare behavior, not only rewards or spike counts.
- Update [MINECRAFT.md](MINECRAFT.md), this roadmap and the local handoff after
  each milestone; publish measured results with their actual limitations.

## Repository About text

Proposed GitHub description (requires authenticated repository-settings access):

> Experimental Drosophila connectome simulator with a live neural viewer and a learning Minecraft fly: RGB eyes, persistent memory, foraging and player-like controls.
