from pathlib import Path

import pandas as pd


CACHE = Path("data/cache")
OUT = CACHE / "motor_map.csv"


def clean(value):
    if pd.isna(value):
        return ""
    return str(value)


def detect_side(row):
    instance = clean(row.get("instance"))
    soma_side = clean(row.get("somaSide"))

    if instance.endswith("_L"):
        return "L"

    if instance.endswith("_R"):
        return "R"

    if soma_side in ("L", "R"):
        return soma_side

    return "?"


def detect_system(row):
    """
    Determine broad motor system primarily from peripheral exit nerve.

    This is much more reliable than trying to infer everything
    from names such as MN2Db, PS349, etc.
    """

    nerve = clean(row.get("exitNerve"))

    # --------------------------------------------------
    # Legs
    # --------------------------------------------------

    if nerve in {
        "ProLN",
        "MesoLN",
        "MetaLN",
    }:
        return "leg"

    # --------------------------------------------------
    # Wings
    # --------------------------------------------------

    if nerve in {
        "ADMN",
        "PDMNa",
        "PDMNp",
        "PDMN",
        "MesoAN",
    }:
        return "wing"

    # --------------------------------------------------
    # Haltere
    # --------------------------------------------------

    if nerve == "DMetaN":
        return "haltere"

    # --------------------------------------------------
    # Abdomen
    # --------------------------------------------------

    if nerve in {
        "AbN1",
        "AbN2",
        "AbN3",
        "AbN4",
        "AbNT",
    }:
        return "abdomen"

    return None


def detect_leg_pair(row):
    """
    Determine front / middle / hind leg.

    Prefer exit nerve, then soma neuromere, then text clues.
    """

    nerve = clean(row.get("exitNerve"))
    neuromere = clean(row.get("somaNeuromere"))

    # Strongest evidence: peripheral nerve

    if nerve == "ProLN":
        return "front"

    if nerve == "MesoLN":
        return "middle"

    if nerve == "MetaLN":
        return "hind"

    # Fallback: soma neuromere

    if neuromere == "T1":
        return "front"

    if neuromere == "T2":
        return "middle"

    if neuromere == "T3":
        return "hind"

    # Final fallback: annotation text

    text = " ".join([
        clean(row.get("instance")),
        clean(row.get("type")),
        clean(row.get("mancType")),
        clean(row.get("entryNerve")),
        clean(row.get("exitNerve")),
    ]).lower()

    if any(x in text for x in [
        "front",
        "prothor",
    ]):
        return "front"

    if any(x in text for x in [
        "middle",
        "mesothor",
    ]):
        return "middle"

    if any(x in text for x in [
        "hind",
        "metathor",
    ]):
        return "hind"

    return "unknown"


def classify(row):

    text = " ".join([
        clean(row.get("instance")),
        clean(row.get("type")),
        clean(row.get("mancType")),
        clean(row.get("synonyms")),
    ]).lower()

    # --------------------------------------------------
    # Broad body system
    # --------------------------------------------------

    nerve_system = detect_system(row)

    if nerve_system is not None:
        limb = nerve_system

    elif "wing" in text:
        limb = "wing"

    elif "haltere" in text:
        limb = "haltere"

    elif any(x in text for x in [
        "neck",
        "head",
    ]):
        limb = "neck"

    elif any(x in text for x in [
        "abdomen",
        "abdominal",
    ]):
        limb = "abdomen"

    elif any(x in text for x in [
        "tibia",
        "ti ",
        " ti_",
        "femur",
        "fe ",
        "coxa",
        "trochanter",
        "sternotrochanter",
        "leg",
    ]):
        limb = "leg"

    else:
        limb = "other"

    # --------------------------------------------------
    # Action
    # --------------------------------------------------

    actions = [
        ("extensor", "extend"),
        ("extension", "extend"),

        ("flexor", "flex"),
        ("flexion", "flex"),

        ("levator", "lift"),
        ("elevator", "lift"),

        ("depressor", "depress"),

        ("protractor", "protract"),
        ("retractor", "retract"),

        ("adductor", "adduct"),
        ("abductor", "abduct"),
    ]

    action = "generic"

    for token, result in actions:
        if token in text:
            action = result
            break

    # --------------------------------------------------
    # Leg segment
    # --------------------------------------------------

    if any(x in text for x in [
        "tibia",
        "ti ",
        " ti_",
    ]):
        segment = "tibia"

    elif any(x in text for x in [
        "femur",
        "fe ",
    ]):
        segment = "femur"

    elif "coxa" in text:
        segment = "coxa"

    elif "trochanter" in text:
        segment = "trochanter"

    else:
        segment = "unknown"

    return limb, segment, action


def main():

    metadata = pd.read_parquet(
        CACHE / "metadata.parquet"
    )

    # --------------------------------------------------
    # Clean body IDs
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Detect motor neurons
    # --------------------------------------------------

    instance = (
        metadata["instance"]
        .fillna("")
        .astype(str)
    )

    superclass = (
        metadata["superclass"]
        .fillna("")
        .astype(str)
    )

    motor = metadata[
        superclass.str.contains(
            "motor",
            case=False,
            regex=False,
        )
        |
        instance.str.contains(
            r"\bMN\b",
            case=False,
            regex=True,
        )
    ].copy()

    # --------------------------------------------------
    # Classify each motor neuron
    # --------------------------------------------------

    rows = []

    for _, row in motor.iterrows():

        limb, segment, action = classify(row)

        if limb == "leg":
            leg_pair = detect_leg_pair(row)
        else:
            leg_pair = ""

        rows.append({
            "bodyId": int(row["bodyId"]),

            "instance": row.get(
                "instance"
            ),

            "type": row.get(
                "type"
            ),

            "side": detect_side(row),

            "limb": limb,

            "leg_pair": leg_pair,

            "segment": segment,

            "action": action,

            "somaNeuromere": row.get(
                "somaNeuromere"
            ),

            "entryNerve": row.get(
                "entryNerve"
            ),

            "exitNerve": row.get(
                "exitNerve"
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

            "synonyms": row.get(
                "synonyms"
            ),
        })

    result = pd.DataFrame(rows)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    result.to_csv(
        OUT,
        index=False,
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("MOTOR MAP")
    print("=" * 80)

    print(
        f"Motor neurons found: "
        f"{len(result):,}"
    )

    print("\nBy system:")
    print(
        result["limb"]
        .value_counts()
        .to_string()
    )

    print("\nLeg pairs:")

    legs = result[
        result["limb"] == "leg"
    ]

    if len(legs):
        print(
            legs["leg_pair"]
            .value_counts()
            .to_string()
        )

    print("\nLeg actions:")

    if len(legs):
        print(
            legs["action"]
            .value_counts()
            .to_string()
        )

    print("\nLeg segments:")

    if len(legs):
        print(
            legs["segment"]
            .value_counts()
            .to_string()
        )

    print("\nExit nerves:")
    print(
        result["exitNerve"]
        .fillna("unknown")
        .value_counts()
        .head(30)
        .to_string()
    )

    print(
        f"\nSaved: {OUT}"
    )

    # --------------------------------------------------
    # Known test neuron
    # --------------------------------------------------

    test = result[
        result["bodyId"] == 815344
    ]

    if len(test):

        print()
        print("=" * 80)
        print("KNOWN TEST MOTOR NEURON")
        print("=" * 80)

        print(
            test.to_string(
                index=False
            )
        )

    # --------------------------------------------------
    # Examples
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("EXAMPLE MOTOR NEURONS")
    print("=" * 80)

    print(
        result.head(40)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
