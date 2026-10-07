"""Paired, multi-episode evaluation of the arena's engineered learning controller."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from src.arena_learning import ExplorationLearner
from src.arena_model import Arena, DEFAULT_ARENA_SIZE
from src.arena_session import ArenaSession
from src.live_brain import load_live_graph


def episode_start(arena, seed, episode, phase):
    """Independent start RNG; evaluation starts do not depend on training length."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode, phase]))
    for _ in range(10000):
        position = rng.uniform([-arena.half_width + 2, -arena.half_height + 2],
                               [arena.half_width - 2, arena.half_height - 2])
        if all(np.linalg.norm(position - [x, y]) > radius + 2
               for x, y, radius in arena.obstacles):
            return position, float(rng.uniform(-np.pi, np.pi))
    raise ValueError("Could not sample an unobstructed episode start")


def run_episode(session, start, steps, seed):
    session.reset()
    position, yaw = start
    session.arena.position = position.copy()
    session.arena.yaw = yaw
    session.arena.gait.reset(position, yaw)
    if session.vision is not None:
        session.vision.reset()
        session.visual_frame = session.vision.sample(session.arena, session.dt)
    if session.learner is not None:
        session.learner.rng = np.random.default_rng(seed)
    visited = {tuple(np.floor(position / .75).astype(int))}
    action_steps = np.zeros(len(ExplorationLearner.names), dtype=int)
    assist_steps = airborne_steps = stuck_steps = 0
    max_altitude = 0.0
    for _ in range(steps):
        session.tick()
        visited.add(tuple(np.floor(session.arena.position / .75).astype(int)))
        assist_steps += int(session.navigator.active)
        airborne_steps += int(session.arena.altitude > .15)
        stuck_steps += int(session.arena.speed < .2)
        max_altitude = max(max_altitude, session.arena.altitude)
        if session.learner is not None:
            action_steps[session.learner.action] += 1
        if session.brain.halted:
            break
    return {
        "start_position": position.tolist(), "start_yaw": yaw,
        "requested_steps": steps, "completed_steps": session.brain.step_count,
        "halted": bool(session.brain.halted), "distance": session.arena.distance,
        "visited_tiles": len(visited), "contact_steps": session.arena.contacts,
        "assist_steps": assist_steps, "assist_interventions": session.navigator.interventions,
        "airborne_steps": airborne_steps, "low_speed_steps": stuck_steps,
        "max_altitude": max_altitude, "motor_spikes": int(session.bridge.motor_spikes),
        "visual_spikes": session.visual_spikes,
        "action_steps": dict(zip(ExplorationLearner.names, action_steps.tolist())),
        "policy_updates": session.learner.updates if session.learner else 0,
    }


METRICS = ("distance", "visited_tiles", "contact_steps", "assist_steps",
           "assist_interventions", "airborne_steps", "low_speed_steps")


def summarize(rows):
    """Keep aborted episodes visible, but exclude them from paired comparisons."""
    groups, paired = [], []
    for assist in sorted({row["assist"] for row in rows}):
        evaluation = [r for r in rows if r["assist"] == assist and r["phase"] == "evaluation"]
        for controller in ("fixed", "untrained", "trained"):
            selected = [r for r in evaluation if r["controller"] == controller]
            complete = [r for r in selected if not r["halted"] and r["completed_steps"] == r["requested_steps"]]
            groups.append({"assist": assist, "controller": controller,
                           "episodes": len(selected), "complete_episodes": len(complete),
                           "means": {key: float(np.mean([r[key] for r in complete]))
                                     for key in METRICS} if complete else {}})
        indexed = {(r["seed"], r["episode"], r["controller"]): r for r in evaluation
                   if not r["halted"] and r["completed_steps"] == r["requested_steps"]}
        for baseline in ("fixed", "untrained"):
            deltas = []
            for seed, episode in sorted({(r["seed"], r["episode"]) for r in evaluation}):
                trained = indexed.get((seed, episode, "trained"))
                reference = indexed.get((seed, episode, baseline))
                if trained is not None and reference is not None:
                    deltas.append({key: trained[key] - reference[key] for key in METRICS})
            paired.append({"assist": assist, "comparison": "trained_minus_" + baseline,
                           "complete_pairs": len(deltas),
                           "mean_deltas": {key: float(np.mean([r[key] for r in deltas]))
                                           for key in METRICS} if deltas else {}})
    return {"groups": groups, "paired": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--train-episodes", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--assist", choices=["both", "on", "off"], default="both")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--arena-width", type=float, default=DEFAULT_ARENA_SIZE[0])
    parser.add_argument("--arena-height", type=float, default=DEFAULT_ARENA_SIZE[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds)
            or min(args.train_episodes, args.eval_episodes, args.steps) < 1
            or not 12 <= args.arena_width <= 2000 or not 12 <= args.arena_height <= 2000):
        parser.error("Use unique nonnegative seeds, positive episode lengths/counts, and room dimensions 12–2000")
    output = args.output or Path("runs") / datetime.now().strftime("arena_evaluation_%Y%m%d_%H%M%S_%f")
    # Refuse reuse so experiments and policy snapshots cannot silently be replaced.
    output.mkdir(parents=True, exist_ok=False)
    size = (args.arena_width, args.arena_height)
    arena = Arena(half_width=size[0]/2, half_height=size[1]/2)
    config = vars(args).copy()
    config["output"] = str(output)
    config.update({"dataset": "male-cns:v1.0", "neural_hz": 60, "vision": True,
                   "initial_policy": "fresh zero Q table for each seed and assist condition",
                   "protocol": "Disjoint training/evaluation starts; identical evaluation starts and action RNG seeds across controllers. Evaluation freezes Q values and counters.",
                   "limitations": "Toy dynamics/kinematics. Fixed drive never requests flight; untrained policy is the matched eight-action control. Episodes within a seed share training; descriptive means are not significance tests. Learning uses existing half-second updates, omitting the final partial reward interval."})
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    graph = load_live_graph()
    session = ArenaSession(graph, device=args.device, learning=False, max_events=0, arena_size=size)
    config["backend"] = str(session.brain.device)
    config["retained_bodies"] = len(graph.ids)
    manifest = Path("data/cache/live_brain_v1/manifest.json")
    config["graph_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = []

    def record(seed, episode, assist, phase, controller, start):
        action_seed = np.random.SeedSequence([seed, episode, int(phase == "evaluation"), 99])
        row = run_episode(session, start, args.steps, action_seed)
        row.update(seed=seed, episode=episode, assist=assist, phase=phase, controller=controller)
        rows.append(row)
        with (output / "episodes.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"assist={assist} seed={seed} {phase}/{controller} episode={episode}: "
              f"tiles={row['visited_tiles']} distance={row['distance']:.2f} "
              f"contacts={row['contact_steps']} halted={row['halted']}", flush=True)

    for assist in ([False, True] if args.assist == "both" else [args.assist == "on"]):
        session.avoidance_enabled = assist
        for seed in args.seeds:
            learner = ExplorationLearner(seed=seed)
            session.learner = learner
            for episode in range(args.train_episodes):
                record(seed, episode, assist, "training", "training",
                       episode_start(arena, seed, episode, 0))
            learner.training = False
            policy_path = output / f"assist_{int(assist)}_seed_{seed}_policy.json"
            learner.save(policy_path)
            frozen = (learner.q.copy(), learner.updates, learner.lifetime_reward)
            for episode in range(args.eval_episodes):
                start = episode_start(arena, seed, episode, 1)
                for controller, policy in (("fixed", None), ("untrained", ExplorationLearner(training=False)),
                                           ("trained", learner)):
                    session.learner = policy
                    record(seed, episode, assist, "evaluation", controller, start)
            if (not np.array_equal(learner.q, frozen[0]) or learner.updates != frozen[1]
                    or learner.lifetime_reward != frozen[2]):
                raise RuntimeError("Evaluation changed learned values or training counters")
    report = summarize(rows)
    report["aborted_episodes"] = sum(row["halted"] for row in rows)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Evaluation saved to {output}", flush=True)
    if report["aborted_episodes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
