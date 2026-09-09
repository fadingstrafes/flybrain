from pathlib import Path
from google.cloud import storage

BUCKET = "flyem-male-cns"
PREFIX = "v1.0/connectome-data/flat-connectome"

FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]

out = Path("data/raw")
out.mkdir(parents=True, exist_ok=True)

client = storage.Client.create_anonymous_client()
bucket = client.bucket(BUCKET)

for name in FILES:
    destination = out / name

    if destination.exists():
        print(f"Already exists: {destination}")
        continue

    print(f"Downloading {name}...")

    blob = bucket.blob(f"{PREFIX}/{name}")
    blob.download_to_filename(destination)

    print(
        f"Saved {destination} "
        f"({destination.stat().st_size / 1024**2:.1f} MiB)"
    )

print("\nDone.")
