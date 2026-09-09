# AGENTS.md

This repository is the FlyBrain project: a local Drosophila Male CNS connectome simulator and realtime viewer.

Before making substantial changes, read `PROJECT_HANDOFF.md`.

## Critical rules

- Preserve `src/sim_full.py` as the CPU reference unless explicitly asked to modify it.
- `W[pre, post]` stores anatomical connection weight from presynaptic neuron to postsynaptic neuron.
- For vectorized propagation from a presynaptic activity vector, use `W.T @ x`.
- Never create a dense whole-CNS matrix.
- Keep connectome math sparse.
- Current NT sign model: ACh `+1`, GABA `-1`, all other transmitters `0`.
- Treat dynamics and motor kinematics as modeling assumptions, not real physiology.
- Never print, hard-code, or commit `NEUPRINT_APPLICATION_CREDENTIALS`.
- Preserve run output compatibility (`spike_events.npz`, `spike_counts.npz`, CSVs, `params.json`) when practical.
- AMD RX 9070 XT uses ROCm/PyTorch; `torch.cuda` APIs are expected.
- Prefer the ModernGL/OpenGL viewer for new realtime rendering work; Matplotlib viewers are fallback/debug tools.
- Do not delete working scripts while prototyping replacements.

## Environment

```bash
cd ~/flybrain
source .venv/bin/activate
```

Project data/cache locations are under `data/raw` and `data/cache`.

Run tests/sanity checks against known IDs such as DNg15_R `513052` and Ti extensor MN_L `815344`.
