from pathlib import Path
import argparse

import numpy as np
import pandas as pd


CACHE = Path("data/cache")


def load_events(run):
    data = np.load(Path(run) / "spike_events.npz")

    df = pd.DataFrame({
        "time": data["time"].astype(int),
        "bodyId": data["bodyId"].astype(int),
    })

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("run_a")
    parser.add_argument("run_b")

    parser.add_argument(
        "--name-a",
        default="A",
    )

    parser.add_argument(
        "--name-b",
        default="B",
    )

    args = parser.parse_args()

    motor = pd.read_csv(
        CACHE / "motor_map.csv"
    )

    motor["bodyId"] = motor[
        "bodyId"
    ].astype(int)

    motor = motor.set_index(
        "bodyId"
    )

    a = load_events(args.run_a)
    b = load_events(args.run_b)

    # Only motor-neuron events
    motor_ids = set(motor.index)

    a = a[
        a["bodyId"].isin(motor_ids)
    ]

    b = b[
        b["bodyId"].isin(motor_ids)
    ]

    counts_a = (
        a.groupby("bodyId")
        .size()
        .rename("spikes_a")
    )

    counts_b = (
        b.groupby("bodyId")
        .size()
        .rename("spikes_b")
    )

    result = pd.concat(
        [counts_a, counts_b],
        axis=1,
    ).fillna(0)

    result["spikes_a"] = (
        result["spikes_a"]
        .astype(int)
    )

    result["spikes_b"] = (
        result["spikes_b"]
        .astype(int)
    )

    result["difference"] = (
        result["spikes_b"]
        - result["spikes_a"]
    )

    # Add motor metadata
    result = result.join(
        motor[
            [
                "instance",
                "side",
                "limb",
                "leg_pair",
                "segment",
                "action",
                "exitNerve",
            ]
        ],
        how="left",
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    active_a = set(
        result[
            result["spikes_a"] > 0
        ].index
    )

    active_b = set(
        result[
            result["spikes_b"] > 0
        ].index
    )

    only_a = active_a - active_b
    only_b = active_b - active_a
    both = active_a & active_b

    print()
    print("=" * 80)
    print("RUN COMPARISON")
    print("=" * 80)

    print(
        f"{args.name_a}: "
        f"{len(active_a)} active motor neurons, "
        f"{result['spikes_a'].sum()} motor spikes"
    )

    print(
        f"{args.name_b}: "
        f"{len(active_b)} active motor neurons, "
        f"{result['spikes_b'].sum()} motor spikes"
    )

    print()
    print(
        f"Only {args.name_a}: {len(only_a)}"
    )

    print(
        f"Only {args.name_b}: {len(only_b)}"
    )

    print(
        f"Active in both: {len(both)}"
    )

    # --------------------------------------------------
    # Motor-system totals
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("BY MOTOR SYSTEM")
    print("=" * 80)

    system_summary = (
        result
        .groupby("limb")[
            ["spikes_a", "spikes_b"]
        ]
        .sum()
        .sort_values(
            "spikes_b",
            ascending=False,
        )
    )

    print(system_summary.to_string())

    # --------------------------------------------------
    # Leg totals
    # --------------------------------------------------

    legs = result[
        result["limb"] == "leg"
    ]

    print()
    print("=" * 80)
    print("LEG OUTPUT")
    print("=" * 80)

    leg_summary = (
        legs
        .groupby(
            ["side", "leg_pair"]
        )[
            ["spikes_a", "spikes_b"]
        ]
        .sum()
    )

    print(
        leg_summary.to_string()
    )

    # --------------------------------------------------
    # Largest differences
    # --------------------------------------------------

    print()
    print("=" * 80)
    print(
        f"MOST INCREASED IN {args.name_b}"
    )
    print("=" * 80)

    print(
        result
        .sort_values(
            "difference",
            ascending=False,
        )
        .head(20)
        [
            [
                "instance",
                "side",
                "limb",
                "leg_pair",
                "segment",
                "action",
                "spikes_a",
                "spikes_b",
                "difference",
            ]
        ]
        .to_string()
    )

    print()
    print("=" * 80)
    print(
        f"MOST INCREASED IN {args.name_a}"
    )
    print("=" * 80)

    print(
        result
        .sort_values(
            "difference",
            ascending=True,
        )
        .head(20)
        [
            [
                "instance",
                "side",
                "limb",
                "leg_pair",
                "segment",
                "action",
                "spikes_a",
                "spikes_b",
                "difference",
            ]
        ]
        .to_string()
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    out = Path("runs") / "comparison.csv"

    result.to_csv(out)

    print()
    print(f"Saved full comparison to {out}")


if __name__ == "__main__":
    main()
