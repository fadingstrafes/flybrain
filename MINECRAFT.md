# FlyBrain for Minecraft 26.3 — experimental first version

This Fabric mod spawns an **independent Survival player named FlyBrain**, using
Carpet's fake-player implementation. You remain your own player. No second
account or second Minecraft client is needed. A procedural voxel fly shell adds
six legs, red eyes, antennae, segmented body and animated wings. It can stand,
crawl and fly, while retaining its own player inventory and block interactions.
The first version supports the owner of a **singleplayer / integrated-server**
world. Dedicated-server administration is not exposed yet.

## Install and start

1. Use Minecraft **26.3**, Fabric Loader **0.19.5 or later**, Java **25**, Fabric
   API for 26.3, and **Carpet 26.3** (`fabric-carpet-26.3+v260915.jar`).
2. Put `minecraft-mod/build/libs/flybrain-minecraft-0.1.0.jar` and Carpet in the
   instance's `mods` directory and restart Minecraft. Do not install a duplicate
   Fabric API if your modpack already includes it. The build uses API 0.162.0.
3. From this repository, start the brain and companion map:

   ```bash
   .venv/bin/python -B -m src.minecraft_service --device cuda --viewer
   ```

4. Enter a Survival test world and run **`/flybrain spawn`**. The fly is its own
   player; it never takes over your character. It starts with four bread and adds
   up to eight ripe berry bushes on nearby suitable empty ground.
5. **F8** pauses/resumes the fly; **F9** switches between your view and following
   the fly. Your player body remains in the world while following; it is not
   teleported or made invulnerable. The camera follows while the entity is loaded
   by your client, and returns to you on death/unload. You can also walk around
   normally to watch it. F3+P can disable pause-on-lost-focus for the brain window.
6. **`/flybrain remove`** removes the agent and fly shell. Closing the world also
   removes it. Learned memory stays in Python. Spawn again to start another body.
7. Close the brain window or Ctrl+C the service to save and stop. Losing the
   connection stops agent actions; start the service and press F8 to resume.

On acknowledged death, the agent respawns at its initial test spawn location.
The ordinary death/drop rules apply. Bed-based respawn and Hardcore are not yet
supported. Agent population is currently one per world/service. The localhost
bridge is fixed to `127.0.0.1:8765` in the mod.

## Brain window

Uses the existing `BrainMap` and `BrainRenderer`: real cached anatomical soma
positions and available neuron branches, driven by this session's actual spikes.
Right-drag orbits; scroll zooms; left-click selects a neuron. H toggles active-only,
L toggles branches, C resets the camera, Escape closes the service and saves.
The title shows the last action, health, food, learned death count, update count,
active neuron count and the selected neuron's spike count in this life.
The default view shows only cells that fired in the latest six-step observation
batch. H switches to all mapped neurons; inactive selected neurons are cleared in
active-only mode. Activity represents neural simulation time, not game seconds.
When input stops, the last batch stays visible with a stale-input banner.

The sidebar shows health, hunger, injury/pain and stress **proxies**, current
estimated drive, action repetition, inventory state, reward pulse and nine depth
rays. Pain = max(previous pain × 0.85, health lost / 10), capped at 1, per
observation. Stress = 0.5 pain + 0.3 health deficit + 0.15 fire + 0.05 collision,
capped at 1. These engineered scores are not decoded subjective emotions.
Transmitter rows count firing cells/spikes grouped by cached NT annotations.
Dopamine-cell firing is separate from the positive, artificial dopamine-like
reward pulse. No chemical concentrations or dopamine/serotonin/octopamine
receptor dynamics are simulated; existing transmitter signs are unchanged.

## What can it do?

There are 35 discrete actions: idle; four movement directions; jump;
forward+jump; sneak; four look directions; attack/break; use/place/eat/interact;
nine hotbar selections; inventory open/close; next/previous container slot;
left-click, right-click and quick-move for that slot; takeoff, land, ascend,
descend, crawl and stand. Container slots include
crafting inputs and outputs. It must discover useful combinations itself.

Movement lasts ten game ticks; attack and use last forty ticks so breaking and
consumption have time to complete. Attack/use are disabled while a logical inventory or container is open.
Movement exits that inventory/container; it no longer silently blocks walking. The first version does not expose sprint, offhand,
recipe-book shortcuts or text/sign editing.

## Learning and memory

Default memory: `data/cache/minecraft/policy.json`. Independent of the existing
arena policy. Checkpoints are written atomically every 100 transitions, on a
learned death, and on clean shutdown. A process lock prevents two services from
writing the same policy. Corrupt/incompatible checkpoints fail loudly.

The anatomical connectome stays fixed. A small linear Q readout learns from 64
pooled CNS spike features plus a bias, 55 explicit sensory/survival/foraging cues
and 32 pools of actual visual-input spikes (152 features total); this is **external artificial learning**,
not validated biological plasticity. Eligibility traces distribute feedback
across recent decisions. A bounded buffer stores 512 transitions and rehearses
up to 64 on death. Weights, replay, counters and RNG state survive restarts;
membrane state, short-term traces and spike counters reset at the next life.
A crash can lose updates since the last checkpoint.

The user requested explicit survival drives after testing the initial reward-only
baseline. These are engineered incentives, not innate instincts recovered from
the connectome. There are no building templates or scripted navigation actions.

Reward per observed transition:

- Eating: +0.1 per hunger point restored, capped by the preceding deficit.
- Injury: −0.15 per health point lost; death: −2, with no positive terminal reward.
- Food acquired: +0.03 to +0.1 per new stock unit, depending on hunger, up to 16.
- Gathering: +0.015 per new block-item stock unit, up to 32.
- Improved cover: +0.1 to +0.3 times the increase, depending on danger; +0.04 per
  placed block on that transition if cover improved, capped at eight blocks.
- Visible hostile exposure: −0.01 × proximity signal.
- Exploration: +0.005 on entering a new four-block cell, up to 4,096 cells per life.
- Inactivity: −0.01 after 200 game ticks without observed movement or progress.
  Short rests and safe recovery with health below 20 and food at least 18 are exempt.

Stock and cover rewards use per-life high-water marks to limit drop/pickup and
cover-entry reward loops. They do not prove causal attribution: material crafting
can increase stock; a roof may be natural. Cover means a roof within six blocks
plus nearby walls, not a guaranteed safe shelter. Building incentives cannot
promise that the learner will discover construction sequences.

Food is any item with Minecraft's FOOD component (including potentially harmful
foods); damage can provide negative experience afterward. Threats are living
Enemy-type entities within 12 blocks and line of sight, plus actual injury and
fire signals. This is a supplied category, not learned visual recognition, and
can include a hostile-type mob that is not currently attacking. Night/dark sky
and rain increase the cover incentive. Minecraft game ticks, stock totals,
placement/breaking counts, position and occupied hotbar slots are also observed.

Inapplicable actions are masked: empty/already-selected hotbar slots, container
clicks outside inventory, redundant posture changes and vertical flight commands
when grounded. Exploration remains random at least 12% of decisions. Weights and
replay from v1/v2 checkpoints are padded to the expanded feature set, retaining
the old weights; a `.v1.backup.json` or `.v2.backup.json` is written before a
training migration. Old experiences retain their original rewards. Updated
policies are version 3.

`--evaluate` freezes weights and disables random exploration, but tie-breaking
still uses RNG. It does not write the policy. Use `--policy` and `--log` for
separate experiments. Longer life or purposeful construction has **not yet been
demonstrated**. This small baseline has limited memory and perceptual resolution;
unrestricted crafting/building mastery is not an expected immediate outcome.

## Senses and assumptions

The mod sends actual RGB eye images plus 32 bounded body/symbolic/depth channels:
constant background input, health deficit, hunger, ground/water/fire, speed,
held-food cue, hotbar selection, container-open state, selected slot position,
slot count/food cue, cursor count, pitch, collision; nine local 8-block depth
rays; five hash channels combining the forward ray block's registry name and
selected inventory item's name; flight and crawl state. Food tags are deliberate prior information.
There is no global map or hidden-block scan. The supplemental threat cue identifies
visible hostile categories; it is not pixel-based recognition. Carpet interactions
can hit/use entities in reach. Hashes can collide and are a crude
object representation. Nearest ray hits constrain depth to visible surfaces.

A fixed seeded mapping assigns these channels to 512 annotated sensory neurons
(16 per channel). It is synthetic and not a claim of retinal, hunger or taste
anatomy. Six sparse neural steps run per observation. Actual spikes feed a fixed
64-bin readout with short-term filtering. Game time and modeled neural time are
not physiologically calibrated. Anatomical matrices remain sparse, with existing
`W.T @ x` propagation and NT signs unchanged. `src/sim_full.py` is untouched.

## Protocol and records

`POST /step` accepts version=1, agent-session UUID, life, monotonically increasing
sequence, death flag, health, food and 32 senses, plus optional bounded world
telemetry for the survival drives. Older mods omit the added cues and rewards. One request is in flight at a
time. Response echoes identity and selects one action, plus respawn acknowledgement.
Retries of the latest observation are idempotent; old sequence/life messages and
recently retired client sessions are rejected. A new session drops incomplete
credit assignment, preserving learned weights. Requests have size and value limits;
browser-origin POSTs are rejected. `GET /status` exposes the last summary.

`runs/minecraft/events.jsonl` records observations, actions and rewards. These are
behavioral logs, not complete spike-event replay exports. Image metadata is logged,
but RGB payloads are omitted to keep logs small; full visual replay is not recorded. No original run formats
or datasets are modified. The new viewer reads live arrays directly.

## Build and validate

With JDK 25 on PATH / JAVA_HOME:

```bash
cd minecraft-mod
./gradlew build
```

From the repository root:

```bash
.venv/bin/python -B -m unittest tests.test_minecraft tests.test_minecraft_drives tests.test_minecraft_vision tests.test_live_arena -v
.venv/bin/python -B -m src.minecraft_smoke --device cuda
```

The smoke check uses synthetic observations against the real sparse graph, an
isolated temporary policy and an offscreen OpenGL render. It never trains the
user's Minecraft policy. It checks real neural response and checkpoint restoration,
not in-game skill. Reports and a brain screenshot go to `runs/minecraft_smoke`.

Validation on 2026-10-07: 23 targeted Python tests and the full 62-test suite
passed with GPU access. The real-image GPU smoke produced 210,179 spikes across
167,176 retained cells, restored a temporary checkpoint and rendered the eye/brain
dashboard. The isolated Minecraft test passed independent spawning, inventory
crafting, Survival block placement/breaking, crawl/flight, damage/death, starter
eating, berry harvesting and paired RGB capture. It also passed with the installed
Fabulously Optimized jars in default test settings. Active shader packs and the
user's existing world/configuration were not tested. The login-time "invalid
player data" regression is covered; fake players acknowledge loading so normal
damage rules apply. These results validate integration, not improved survival.

## Fly embodiment

The underlying player has normal Survival health/hunger, inventory, collision and
item/block interactions. The fly is intentionally player-sized, not insect-sized.
The model has a stepped segmented body, faceted red eyes, antennae, six bent
two-segment legs with cosmetic walking motion, and translucent veined wings.
Wings lie along the back while standing and animate in flight. It is a cosmetic
collection of vanilla block-display entities, tagged `flybrain_visual`, removed
with the agent and recreated on dimension changes. Vanilla player rendering is
suppressed by a client renderer hook for the agent's UUID, including human
armor/item layers. Server metadata cannot overwrite this hook as it could the
old visibility flag that caused Steve flicker. Server-side invisibility is not
enabled. Other players render normally.
Crawling uses the player's low swimming pose; standing restores ordinary pose.
Flight deliberately grants flight ability in Survival, with an artificial hunger
exhaustion cost of 0.03 per running server tick. Vertical actions select ascent,
descent or hovering. These are engineered body rules, not realistic fly mechanics.
Damage reduces real game health and produces a negative learning signal; no claim
of subjective pain is made. This is player-like capability, not a guarantee of
complete equivalence to a connected human client.

The earlier client-takeover prototype is retained as `PlayerControlPrototype.java`
for reference but is **not registered or active**.

## Isolated Minecraft integration test

With a desktop display and Java 25:

```bash
cd minecraft-mod
./gradlew runClientGameTest
```

This creates a separate test world and checks independent player spawning,
primitive inventory crafting, block placement and breaking with its own inventory,
crawl/flight controls, damage and death notification.
It also captures an observer screenshot. Scripted test actions validate capability;
they are not examples of learned building and do not train the user's policy.

Carpet dependency: https://modrinth.com/mod/carpet/version/26.3
Official Fabric test documentation: https://docs.fabricmc.net/develop/automatic-testing

## Injury circuits versus state estimates

Real flies have experimentally studied nociceptive pathways. For example,
[adult fly injury/sensitization research](https://pmc.ncbi.nlm.nih.gov/articles/PMC6620091/)
identifies ppk-positive sensory neurons and TrpA1-dependent responses.
[Defensive-arousal experiments](https://www.janelia.org/publication/behavioral-responses-repetitive-visual-threat-stimulus-express-persistent-state)
report persistent responses to repeated visual threats. These do not make an
arbitrary neuron's firing a calibrated pain or fear measurement.

The local type/instance/subclass annotation search found no direct nociceptor,
ppk or painless matches. This does not establish that the underlying cells are
absent; mapping requires stronger identity evidence. Current game damage still
uses the documented synthetic sensory mapping. The viewer therefore reports
actual neural activity and separately labeled game-derived estimates. It does
not claim to have validated a nociception circuit or a fear decoder.

## RGB eyes and visual neurons

Two independent offscreen Minecraft cameras render 96×64 RGB images at up to four
pairs per second. They sit ±0.12 blocks to either side of the player's eye point,
face ±55° from its heading, and use a 100° vertical field of view. They show actual
world textures, lighting, plants and entities. The agent's cosmetic fly shell and
human avatar are excluded from these views. The observer camera is restored after
each capture. The nine depth rays remain additional sensors. Vision is available
only for client-loaded terrain; it is not a hidden-world scan. Capture rate is
limited and does not match real fly temporal resolution. Active shader packs have
not been validated.

The brain window's lower strip shows left/right RGB, compound samples, and
absolute brightness change (displayed ×3). These are the same images/samples used
for stimulation, not a decorative preview. Missing images show a waiting message;
frames older than two seconds are omitted by the mod. Pausing the game freezes
input and the dashboard reports its observation age.

Each eye is resampled onto 768 staggered angular samples. Local annotations map
left/right R1–R6, R7 and R8 populations: **6,091 retained photoreceptors**. The
facet-to-cell assignment is seeded and synthetic, not established retinotopy.
Luminance feeds R1–R6; blue/green are engineering surrogates for R7/R8. Minecraft
RGB provides neither real UV nor polarization information. These approximations
must not be described as a complete real fly eye.

All mapped photoreceptor outgoing signs are zero under the project's existing
transmitter model. To make image signals reach the network, an explicit parallel
image input stimulates **3,551 annotated L2/L3 relays**. This substitutes for a
missing retinal transmission model; it does not change transmitter signs or
connectome weights. Direct visual input totals 9,642 cells, and subsequent
propagation remains sparse. The dashboard distinguishes input-cell firing from
the whole neural population. Its 32 visual readout features pool actual spikes
from those input cells; they are not object-recognition labels.

Validation includes an image-only experiment with body inputs disabled: dark
input produced zero spikes; an illuminated input produced 25,375 direct visual
input spikes and 687 downstream spikes in six steps. This establishes signal
propagation, not object understanding or improved survival. Real game eye captures
and dashboard validation are under `runs/minecraft_eyes_check`.

## Starter food, renewable foraging and shelter quality

A manual `/flybrain spawn` gives four bread and adds up to eight mature sweet
berry bushes within roughly six blocks, only in air over suitable grass/dirt.
It never replaces existing blocks. Bushes stay in the world and use vanilla
harvesting/regrowth; they can also hurt an agent walking into them. Automatic
respawns do not grant another starter kit. `/flybrain forage` adds bushes around
an existing fly on suitable empty ground, without instantly ripening old plants.

Food opportunities include visible edible dropped items, ripe berries, mature
carrots/potatoes/beetroot, and melons within eight blocks. Line-of-sight and broad
view-angle checks gate these **supplied food categories**, separate from the RGB
neural input. Proximity and left/right bearing supplement the readout. Selecting
food and using it, approaching/picking up drops, using berries, and breaking
crops are still actions the learner must choose; there is no auto-forager.

Cover quality requires a roof within six blocks. Its score adds 0.35 for the
roof, 0.1 per blocked cardinal ray (up to three), 0.15 for a nearby standable
exit with at least one open cardinal ray, and up to 0.2 for block light. It
subtracts 0.5 for local lava/fire/magma/submersion and up to 0.4 for visible
hostile proximity, clamped to [0,1]. This favors roofed, lit, partially enclosed,
accessible locations. It is a local approximation: doors, diagonal exits,
structural durability and hidden threats are not fully assessed. The cover
reward encourages improvements, not a particular building template.

The isolated game test verifies starter eating, ripe-bush detection and harvesting
that leaves a regrowing plant, RGB capture including sky, and previous player
capabilities. It also passed using the installed Fabulously Optimized mod jars,
including Sodium/Iris, in a separate test world with default test settings.

## Next development session

Start with repeatable survival evaluation before changing rewards or expanding
the learner. Then test visual contribution with controlled ablations, improve
sustainable foraging, and tackle multi-step gathering/crafting/shelter behavior.
The ordered tasks and acceptance checks are in [ROADMAP.md](ROADMAP.md).
Keep evaluation worlds and policies separate from normal play; the default
learning file is `data/cache/minecraft/policy.json` and event log is
`runs/minecraft/events.jsonl`. Current policy schema is v3 with 152 features;
migrations preserve older weights and back up the previous checkpoint.
