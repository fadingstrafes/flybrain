from pathlib import Path

import pandas as pd


CACHE = Path("data/cache")
OUT = CACHE / "sensory_map.csv"


def clean(value):
    if pd.isna(value):
        return ""
    return str(value)


def detect_side(row):
    instance = clean(row.get("instance"))
    soma_side = clean(row.get("somaSide"))
    root_side = clean(row.get("rootSide"))

    if instance.endswith("_L"):
        return "L"

    if instance.endswith("_R"):
        return "R"

    if soma_side in ("L", "R"):
        return soma_side

    if root_side in ("L", "R"):
        return root_side

    return "?"


def detect_system(row):
    nerve = clean(row.get("entryNerve"))

    # Leg sensory nerves
    if nerve == "ProLN":
        return "front_leg"

    if nerve == "MesoLN":
        return "middle_leg"

    if nerve == "MetaLN":
        return "hind_leg"

    # Wing-associated nerves
    if nerve in {
        "ADMN",
        "PDMNa",
        "PDMNp",
        "PDMN",
        "MesoAN",
    }:
        return "wing"

    if nerve == "DMetaN":
        return "haltere"

    if nerve in {
        "AbN1",
        "AbN2",
        "AbN3",
        "AbN4",
        "AbNT",
    }:
        return "abdomen"

    return "other"


def main():
    metadata = pd.read_parquet(
        CACHE / "metadata.parquet"
    )

    metadata["bodyId"] = pd.to_numeric(
        metadata["bodyId"],
        errors="coerce",
    )

    metadata = (
        metadata
        .dropna(subset=["bodyId"])
        .drop_duplicates("bodyId")
    )

    metadata["bodyId"] = (
        metadata["bodyId"]
        .astype("int64")
    )

    superclass = (
        metadata["superclass"]
        .fillna("")
        .astype(str)
    )

    # Do not guess the exact MaleCNS superclass spelling.
    # Accept anything that contains "sensory".
    sensory = metadata[
        superclass.str.contains(
            "sensory",
            case=False,
            regex=False,
        )
    ].copy()

    rows = []

    for _, row in sensory.iterrows():
        rows.append({
            "bodyId": int(row["bodyId"]),
            "instance": row.get("instance"),
            "type": row.get("type"),
            "side": detect_side(row),
            "system": detect_system(row),
            "entryNerve": row.get("entryNerve"),
            "exitNerve": row.get("exitNerve"),
            "somaNeuromere": row.get(
                "somaNeuromere"
            ),
            "superclass": row.get(
                "superclass"
            ),
            "subclass": row.get(
                "subclass"
            ),
            "mancType": row.get(
                "mancType"
            ),
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT,
        index=False,
    )

    print()
    print("=" * 80)
    print("SENSORY MAP")
    print("=" * 80)

    print(
        f"Sensory neurons found: {len(result):,}"
    )

    if result.empty:
        print(
            "\nNo superclass containing 'sensory' "
            "was found."
        )

        print(
            "\nSuperclass values in metadata:"
        )

        print(
            metadata["superclass"]
            .fillna("None")
            .value_counts()
            .head(50)
            .to_string()
        )

        return

    print("\nBy system:")
    print(
        result["system"]
        .value_counts()
        .to_string()
    )

    print("\nEntry nerves:")
    print(
        result["entryNerve"]
        .fillna("unknown")
        .value_counts()
        .head(40)
        .to_string()
    )

    print("\nBy side:")
    print(
        result["side"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nSaved: {OUT}"
    )

    print()
    print("=" * 80)
    print("FRONT-LEG SENSORY EXAMPLES")
    print("=" * 80)

    front = result[
        result["system"] == "front_leg"
    ]

    print(
        front.head(40)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
